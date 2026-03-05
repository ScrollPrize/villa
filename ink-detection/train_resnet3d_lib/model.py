import os.path as osp
from dataclasses import asdict, is_dataclass
from types import SimpleNamespace

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from metrics import StreamingBinarySegmentationMetrics
from models.resnetall import generate_model
from train_resnet3d_lib.config import CFG, log
from train_resnet3d_lib.modeling.architecture import Decoder, replace_batchnorm_with_groupnorm
from train_resnet3d_lib.modeling.group_dro import GroupDROComputer
from train_resnet3d_lib.modeling.losses import build_bce_targets, compute_per_sample_loss_and_dice
from train_resnet3d_lib.modeling.optimizers_runtime import configure_optimizers as configure_optimizers_runtime
from train_resnet3d_lib.stitch_manager import StitchManager, coerce_stitch_manager_state

def _cfg_to_dict(value, *, key):
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return dict(asdict(value))
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"{key} must be a dict/dataclass/object with attributes, got {type(value).__name__}")


def _coerce_flat_model_state(state):
    data = _cfg_to_dict(state, key="model_state")
    n_groups = int(data.get("n_groups", 1) or 1)
    group_names = data.get("group_names")
    if group_names is None:
        group_names = [str(i) for i in range(n_groups)]
    else:
        group_names = [str(x) for x in group_names]
    stitch_group_idx_by_segment = {
        str(segment_id): int(group_idx)
        for segment_id, group_idx in dict(data.get("stitch_group_idx_by_segment") or {}).items()
    }

    return {
        "size": int(data.get("size", 256)),
        "enc": str(data.get("enc", "i3d")),
        "with_norm": bool(data.get("with_norm", False)),
        "total_steps": int(data.get("total_steps", 1)),
        "n_groups": n_groups,
        "group_names": group_names,
        "stitch_group_idx_by_segment": stitch_group_idx_by_segment,
        "norm": str(data.get("norm", "batch")),
        "group_norm_groups": int(data.get("group_norm_groups", 32)),
        "objective": str(data.get("objective", "erm")),
        "loss_mode": str(data.get("loss_mode", "batch")),
        "loss_recipe": str(data.get("loss_recipe", "dice_bce")).lower(),
        "bce_smooth_factor": float(data.get("bce_smooth_factor", 0.25)),
        "soft_label_positive": float(data.get("soft_label_positive", 1.0)),
        "soft_label_negative": float(data.get("soft_label_negative", 0.0)),
        "robust_step_size": data.get("robust_step_size"),
        "group_counts": [int(x) for x in list(data.get("group_counts") or [])],
        "group_dro_gamma": float(data.get("group_dro_gamma", 0.1)),
        "group_dro_btl": bool(data.get("group_dro_btl", False)),
        "group_dro_alpha": data.get("group_dro_alpha"),
        "group_dro_normalize_loss": bool(data.get("group_dro_normalize_loss", False)),
        "group_dro_min_var_weight": float(data.get("group_dro_min_var_weight", 0.0)),
        "group_dro_adj": data.get("group_dro_adj"),
        "erm_group_topk": int(data.get("erm_group_topk", 0)),
        **coerce_stitch_manager_state(data),
    }


def _coerce_regression_model_state(
    *,
    model_state=None,
):
    if model_state is None:
        return _coerce_flat_model_state({})
    return _coerce_flat_model_state(_cfg_to_dict(model_state, key="model_state"))


def save_regression_hyperparameters(model, *, state):
    model.save_hyperparameters(
        {
            "model_state": {
                "size": int(state.size),
                "enc": str(state.enc),
                "with_norm": bool(state.with_norm),
                "total_steps": int(state.total_steps),
                "n_groups": int(state.n_groups),
                "group_names": list(state.group_names),
                "norm": str(state.norm),
                "group_norm_groups": int(state.group_norm_groups),
                "objective": str(state.objective),
                "loss_mode": str(state.loss_mode),
                "loss_recipe": str(state.loss_recipe),
                "bce_smooth_factor": float(state.bce_smooth_factor),
                "soft_label_positive": float(state.soft_label_positive),
                "soft_label_negative": float(state.soft_label_negative),
                "robust_step_size": state.robust_step_size,
                "group_counts": list(state.group_counts),
                "group_dro_gamma": float(state.group_dro_gamma),
                "group_dro_btl": bool(state.group_dro_btl),
                "group_dro_alpha": state.group_dro_alpha,
                "group_dro_normalize_loss": bool(state.group_dro_normalize_loss),
                "group_dro_min_var_weight": float(state.group_dro_min_var_weight),
                "erm_group_topk": int(state.erm_group_topk),
                "stitch_all_val": bool(state.stitch_all_val),
                "stitch_downsample": int(state.stitch_downsample),
                "stitch_log_only_downsample": int(state.stitch_log_only_downsample),
                "stitch_log_only_every_n_epochs": int(state.stitch_log_only_every_n_epochs),
                "stitch_train": bool(state.stitch_train),
                "stitch_train_every_n_epochs": int(state.stitch_train_every_n_epochs),
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


def _init_group_dro_if_needed(model, *, state):
    model.group_dro = None
    if model.objective != "group_dro":
        return
    robust_step_size = state.robust_step_size
    if robust_step_size is None:
        raise ValueError("group_dro.robust_step_size is required when training.objective is group_dro")
    group_counts = state.group_counts
    if group_counts is None:
        raise ValueError("group_counts is required when training.objective is group_dro")

    model.group_dro = GroupDROComputer(
        n_groups=model.n_groups,
        group_counts=group_counts,
        alpha=state.group_dro_alpha,
        gamma=state.group_dro_gamma,
        adj=state.group_dro_adj,
        min_var_weight=state.group_dro_min_var_weight,
        step_size=robust_step_size,
        normalize_loss=state.group_dro_normalize_loss,
        btl=state.group_dro_btl,
    )


def _build_backbone_with_optional_pretrained(*, state):
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

    norm = str(state.norm).lower()
    group_norm_groups = int(state.group_norm_groups)
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


def _build_stitch_manager(state):
    return StitchManager(**coerce_stitch_manager_state(_cfg_to_dict(state, key="model_state")))


def initialize_regression_state(model, *, state):
    model.objective = state.objective.lower()
    model.loss_mode = state.loss_mode.lower()
    model.loss_recipe = state.loss_recipe.lower()
    model.bce_smooth_factor = state.bce_smooth_factor
    model.soft_label_positive = state.soft_label_positive
    model.soft_label_negative = state.soft_label_negative
    model.with_norm = state.with_norm
    model.total_steps = state.total_steps

    model.n_groups = int(state.n_groups)
    model.group_names = list(state.group_names)
    if len(model.group_names) == 0:
        model.group_names = [str(i) for i in range(model.n_groups)]
    if len(model.group_names) != model.n_groups:
        raise ValueError(f"group_names length must be {model.n_groups}, got {len(model.group_names)}")
    model._stitch_group_idx_by_segment = _normalize_stitch_group_idx_by_segment(
        state.stitch_group_idx_by_segment,
        n_groups=model.n_groups,
    )

    _init_group_dro_if_needed(model, state=state)

    model.erm_group_topk = int(state.erm_group_topk or 0)
    if model.erm_group_topk < 0:
        raise ValueError(f"erm_group_topk must be >= 0, got {model.erm_group_topk}")

    model._ema_decay = float(getattr(CFG, "ema_decay", 0.9))
    model._ema_metrics = {}

    model._eval_threshold = float(getattr(CFG, "eval_threshold", 0.5))
    model._val_eval_metrics = None

    model.loss_func1 = smp.losses.DiceLoss(mode="binary")

    model.backbone = _build_backbone_with_optional_pretrained(state=state)

    norm = str(state.norm).lower()
    group_norm_groups = int(state.group_norm_groups)
    encoder_dims = _infer_encoder_dims(model.backbone)

    model.decoder = Decoder(encoder_dims=encoder_dims, upscale=1, norm=norm, group_norm_groups=group_norm_groups)

    if model.with_norm:
        if norm == "group":
            model.normalization = nn.GroupNorm(num_groups=1, num_channels=1)
        else:
            model.normalization = nn.BatchNorm3d(num_features=1)

    model._stitcher = _build_stitch_manager(state)


def reset_train_epoch_accumulators(model):
    device = model.device
    model._train_loss_sum = torch.tensor(0.0, device=device)
    model._train_dice_sum = torch.tensor(0.0, device=device)
    model._train_count = torch.tensor(0.0, device=device)

    model._train_group_loss_sum = torch.zeros(model.n_groups, device=device)
    model._train_group_dice_sum = torch.zeros(model.n_groups, device=device)
    model._train_group_count = torch.zeros(model.n_groups, device=device)


def accumulate_train_stats(model, per_sample_loss, per_sample_dice, group_idx):
    with torch.no_grad():
        loss_det = per_sample_loss.detach()
        dice_det = per_sample_dice.detach()
        group_idx = group_idx.long()

        model._train_loss_sum += loss_det.sum()
        model._train_dice_sum += dice_det.sum()
        model._train_count += float(loss_det.numel())

        model._train_group_loss_sum.scatter_add_(0, group_idx, loss_det)
        model._train_group_dice_sum.scatter_add_(0, group_idx, dice_det)
        model._train_group_count.scatter_add_(
            0,
            group_idx,
            torch.ones_like(loss_det, dtype=model._train_group_count.dtype),
        )


def compute_group_avg(model, values, group_idx):
    group_idx = group_idx.long()
    group_map = (
        group_idx
        == torch.arange(model.n_groups, device=group_idx.device).unsqueeze(1).long()
    ).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()
    group_avg = (group_map @ values.view(-1)) / group_denom
    return group_avg, group_count


def update_ema_metric(model, name, value):
    decay = float(model._ema_decay)
    if torch.is_tensor(value):
        val = float(value.detach().cpu().item())
    else:
        val = float(value)
    prev = model._ema_metrics.get(name)
    if prev is None:
        ema = val
    else:
        ema = decay * prev + (1.0 - decay) * val
    model._ema_metrics[name] = ema
    model.log(f"{name}_ema", ema, on_step=False, on_epoch=True, prog_bar=False)


def distributed_world_size(model):
    trainer = getattr(model, "trainer", None)
    if trainer is None:
        return 1
    return int(getattr(trainer, "world_size", 1) or 1)


def reduce_sum_distributed(model, tensor):
    if distributed_world_size(model) <= 1:
        return tensor
    strategy = getattr(getattr(model, "trainer", None), "strategy", None)
    if strategy is None or not hasattr(strategy, "reduce"):
        raise RuntimeError("distributed validation reduction requested but trainer.strategy.reduce is unavailable")
    return strategy.reduce(tensor, reduce_op="sum")


def sync_validation_accumulators(model):
    if distributed_world_size(model) <= 1:
        return
    model._val_loss_sum = reduce_sum_distributed(model, model._val_loss_sum)
    model._val_dice_sum = reduce_sum_distributed(model, model._val_dice_sum)
    model._val_bce_sum = reduce_sum_distributed(model, model._val_bce_sum)
    model._val_dice_loss_sum = reduce_sum_distributed(model, model._val_dice_loss_sum)
    model._val_count = reduce_sum_distributed(model, model._val_count)
    model._val_group_loss_sum = reduce_sum_distributed(model, model._val_group_loss_sum)
    model._val_group_dice_sum = reduce_sum_distributed(model, model._val_group_dice_sum)
    model._val_group_bce_sum = reduce_sum_distributed(model, model._val_group_bce_sum)
    model._val_group_dice_loss_sum = reduce_sum_distributed(model, model._val_group_dice_loss_sum)
    model._val_group_count = reduce_sum_distributed(model, model._val_group_count)


def _log_group_train_metrics(model, group_loss, group_count, *, include_adv_probs):
    present = group_count > 0
    if present.any():
        worst_group_loss = group_loss[present].max()
    else:
        worst_group_loss = group_loss.max()
    model.log("train/worst_group_loss", worst_group_loss, on_step=True, on_epoch=False, prog_bar=False)

    for group_i, group_name in enumerate(model.group_names):
        safe_group_name = str(group_name).replace("/", "_")
        model.log(
            f"train/group_{group_i}_{safe_group_name}/loss",
            group_loss[group_i],
            on_step=True,
            on_epoch=False,
        )
        model.log(
            f"train/group_{group_i}_{safe_group_name}/count",
            group_count[group_i],
            on_step=True,
            on_epoch=False,
        )
        if include_adv_probs:
            model.log(
                f"train/group_{group_i}_{safe_group_name}/adv_prob",
                model.group_dro.adv_probs[group_i],
                on_step=True,
                on_epoch=False,
            )


def compute_objective_loss(
    model,
    *,
    outputs,
    targets,
    per_sample_loss,
    per_sample_dice,
    per_sample_bce,
    per_sample_dice_loss,
    group_idx,
):
    objective = str(model.objective).lower()
    loss_mode = str(model.loss_mode).lower()
    loss_recipe = str(getattr(model, "loss_recipe", "dice_bce")).lower()
    group_idx = group_idx.long()

    if objective == "group_dro":
        if model.group_dro is None:
            raise RuntimeError("GroupDRO objective was set but group_dro computer was not initialized")

        robust_loss, group_loss, group_count, _weights = model.group_dro.loss(per_sample_loss, group_idx)
        model.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
        model.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
        model.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)

        if model.global_step % CFG.print_freq == 0:
            _log_group_train_metrics(
                model,
                group_loss,
                group_count,
                include_adv_probs=True,
            )

        return robust_loss

    if loss_mode == "batch":
        bce_targets = model.build_bce_targets(targets)
        bce_loss = F.binary_cross_entropy_with_logits(outputs, bce_targets)
        if loss_recipe == "bce_only":
            dice_loss = per_sample_dice_loss.mean()
            loss = bce_loss
        else:
            dice_loss = model.loss_func1(outputs, targets)
            loss = 0.5 * dice_loss + 0.5 * bce_loss
        model.log("train/dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=False)
        model.log("train/bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    if model.erm_group_topk > 0:
        group_loss, group_count = compute_group_avg(model, per_sample_loss, group_idx)
        present = group_count > 0
        if present.any():
            present_losses = group_loss[present]
            topk = min(int(model.erm_group_topk), int(present_losses.numel()))
            topk_losses, _ = torch.topk(present_losses, topk, largest=True)
            loss = topk_losses.mean()
        else:
            loss = per_sample_loss.mean()

        if model.global_step % CFG.print_freq == 0:
            _log_group_train_metrics(
                model,
                group_loss,
                group_count,
                include_adv_probs=False,
            )
    else:
        loss = per_sample_loss.mean()

    model.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
    model.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
    model.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)
    return loss


def finalize_training_batch(model, *, loss, per_sample_loss, per_sample_dice, group_idx):
    accumulate_train_stats(model, per_sample_loss, per_sample_dice, group_idx)
    if torch.isnan(loss).any():
        raise FloatingPointError("NaN loss encountered during training_step")
    model.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)


def log_train_epoch_metrics(model):
    if model._train_count.item() > 0:
        avg_loss = model._train_loss_sum / model._train_count
        avg_dice = model._train_dice_sum / model._train_count
    else:
        avg_loss = torch.tensor(0.0, device=model.device)
        avg_dice = torch.tensor(0.0, device=model.device)

    group_count = model._train_group_count
    group_loss = model._train_group_loss_sum / group_count.clamp_min(1)
    group_dice = model._train_group_dice_sum / group_count.clamp_min(1)
    worst_group_loss = group_loss.max() if group_loss.numel() else torch.tensor(0.0, device=model.device)

    model.log("train/epoch_avg_loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
    model.log("train/epoch_avg_dice", avg_dice, on_step=False, on_epoch=True, prog_bar=False)
    update_ema_metric(model, "train/total_loss", avg_loss)
    update_ema_metric(model, "train/dice", avg_dice)
    update_ema_metric(model, "train/worst_group_loss", worst_group_loss)
    for group_i, group_name in enumerate(model.group_names):
        safe_group_name = str(group_name).replace("/", "_")
        model.log(
            f"train/group_{group_i}_{safe_group_name}/epoch_loss",
            group_loss[group_i],
            on_step=False,
            on_epoch=True,
        )
        model.log(
            f"train/group_{group_i}_{safe_group_name}/epoch_dice",
            group_dice[group_i],
            on_step=False,
            on_epoch=True,
        )
        model.log(
            f"train/group_{group_i}_{safe_group_name}/epoch_count",
            group_count[group_i],
            on_step=False,
            on_epoch=True,
        )


def reset_validation_epoch_accumulators(model):
    device = model.device
    model._val_loss_sum = torch.tensor(0.0, device=device)
    model._val_dice_sum = torch.tensor(0.0, device=device)
    model._val_bce_sum = torch.tensor(0.0, device=device)
    model._val_dice_loss_sum = torch.tensor(0.0, device=device)
    model._val_count = torch.tensor(0.0, device=device)

    model._val_group_loss_sum = torch.zeros(model.n_groups, device=device)
    model._val_group_dice_sum = torch.zeros(model.n_groups, device=device)
    model._val_group_bce_sum = torch.zeros(model.n_groups, device=device)
    model._val_group_dice_loss_sum = torch.zeros(model.n_groups, device=device)
    model._val_group_count = torch.zeros(model.n_groups, device=device)


def initialize_validation_metrics(model):
    model._val_eval_metrics = StreamingBinarySegmentationMetrics(
        threshold=model._eval_threshold,
        device=model.device,
    )


def accumulate_validation_stats(
    model,
    *,
    per_sample_loss,
    per_sample_dice,
    per_sample_bce,
    per_sample_dice_loss,
    group_idx,
):
    model._val_loss_sum += per_sample_loss.sum()
    model._val_dice_sum += per_sample_dice.sum()
    model._val_bce_sum += per_sample_bce.sum()
    model._val_dice_loss_sum += per_sample_dice_loss.sum()
    model._val_count += float(per_sample_loss.numel())

    group_idx = group_idx.long()
    model._val_group_loss_sum.scatter_add_(0, group_idx, per_sample_loss)
    model._val_group_dice_sum.scatter_add_(0, group_idx, per_sample_dice)
    model._val_group_bce_sum.scatter_add_(0, group_idx, per_sample_bce)
    model._val_group_dice_loss_sum.scatter_add_(0, group_idx, per_sample_dice_loss)
    model._val_group_count.scatter_add_(0, group_idx, torch.ones_like(per_sample_loss, dtype=model._val_group_count.dtype))


def update_validation_stream_metrics(model, *, outputs, targets):
    if model._val_eval_metrics is not None:
        model._val_eval_metrics.update(logits=outputs, targets=targets)


def log_validation_epoch_metrics(model):
    if model._val_count.item() > 0:
        avg_loss = model._val_loss_sum / model._val_count
        avg_dice = model._val_dice_sum / model._val_count
        avg_bce = model._val_bce_sum / model._val_count
        avg_dice_loss = model._val_dice_loss_sum / model._val_count
    else:
        avg_loss = torch.tensor(0.0, device=model.device)
        avg_dice = torch.tensor(0.0, device=model.device)
        avg_bce = torch.tensor(0.0, device=model.device)
        avg_dice_loss = torch.tensor(0.0, device=model.device)

    group_count = model._val_group_count
    group_loss = model._val_group_loss_sum / group_count.clamp_min(1)
    group_dice = model._val_group_dice_sum / group_count.clamp_min(1)
    group_bce = model._val_group_bce_sum / group_count.clamp_min(1)
    group_dice_loss = model._val_group_dice_loss_sum / group_count.clamp_min(1)

    present = group_count > 0
    if present.any():
        worst_group_loss = group_loss[present].max()
        worst_group_dice = group_dice[present].min()
    else:
        worst_group_loss = group_loss.max()
        worst_group_dice = group_dice.min()

    model.log("val/avg_loss", avg_loss, on_epoch=True, prog_bar=True)
    model.log("val/worst_group_loss", worst_group_loss, on_epoch=True, prog_bar=True)
    model.log("val/avg_dice", avg_dice, on_epoch=True, prog_bar=False)
    model.log("val/worst_group_dice", worst_group_dice, on_epoch=True, prog_bar=False)
    model.log("val/avg_bce_loss", avg_bce, on_epoch=True, prog_bar=False)
    model.log("val/avg_dice_loss", avg_dice_loss, on_epoch=True, prog_bar=False)
    update_ema_metric(model, "val/avg_loss", avg_loss)
    update_ema_metric(model, "val/worst_group_loss", worst_group_loss)
    update_ema_metric(model, "val/avg_dice", avg_dice)
    update_ema_metric(model, "val/worst_group_dice", worst_group_dice)

    if model._val_eval_metrics is not None:
        eval_metrics = model._val_eval_metrics.compute()
        for metric_name, metric_value in eval_metrics.items():
            model.log(f"metrics/val/{metric_name}", metric_value, on_epoch=True, prog_bar=False)

    for group_i, group_name in enumerate(model.group_names):
        safe_group_name = str(group_name).replace("/", "_")
        model.log(f"val/group_{group_i}_{safe_group_name}/loss", group_loss[group_i], on_epoch=True)
        model.log(f"val/group_{group_i}_{safe_group_name}/dice", group_dice[group_i], on_epoch=True)
        model.log(f"val/group_{group_i}_{safe_group_name}/bce_loss", group_bce[group_i], on_epoch=True)
        model.log(f"val/group_{group_i}_{safe_group_name}/dice_loss", group_dice_loss[group_i], on_epoch=True)
        model.log(f"val/group_{group_i}_{safe_group_name}/count", group_count[group_i], on_epoch=True)


class RegressionPLModel(pl.LightningModule):
    def __init__(
        self,
        *,
        model_state: dict | None = None,
    ):
        super(RegressionPLModel, self).__init__()
        state = SimpleNamespace(**_coerce_regression_model_state(
            model_state=model_state,
        ))
        save_regression_hyperparameters(
            self,
            state=state,
        )
        initialize_regression_state(
            self,
            state=state,
        )

    def set_stitch_borders(self, *, train_borders=None, val_borders=None):
        self._stitcher.set_borders(train_borders=train_borders, val_borders=val_borders)

    def set_train_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_train_loaders(loaders, segment_ids)

    def set_log_only_stitch_loaders(self, loaders, segment_ids):
        self._stitcher.set_log_only_loaders(loaders, segment_ids)

    def on_train_epoch_start(self):
        reset_train_epoch_accumulators(self)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask

    def compute_per_sample_loss_and_dice(self, logits, targets):
        return compute_per_sample_loss_and_dice(
            logits,
            targets,
            loss_recipe=self.loss_recipe,
            smooth_factor=self.bce_smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def build_bce_targets(self, targets):
        return build_bce_targets(
            targets,
            smooth_factor=self.bce_smooth_factor,
            soft_label_positive=self.soft_label_positive,
            soft_label_negative=self.soft_label_negative,
        )

    def training_step(self, batch, batch_idx):
        x, y, group_idx = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(
            outputs,
            y,
        )

        loss = compute_objective_loss(
            self,
            outputs=outputs,
            targets=y,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            per_sample_bce=per_sample_bce,
            per_sample_dice_loss=per_sample_dice_loss,
            group_idx=group_idx,
        )

        finalize_training_batch(
            self,
            loss=loss,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            group_idx=group_idx,
        )
        return {"loss": loss}

    def on_train_epoch_end(self):
        log_train_epoch_metrics(self)

    def on_validation_epoch_start(self):
        reset_validation_epoch_accumulators(self)
        initialize_validation_metrics(self)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, xyxys, group_idx = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(
            outputs,
            y,
        )

        accumulate_validation_stats(
            self,
            per_sample_loss=per_sample_loss,
            per_sample_dice=per_sample_dice,
            per_sample_bce=per_sample_bce,
            per_sample_dice_loss=per_sample_dice_loss,
            group_idx=group_idx,
        )
        update_validation_stream_metrics(self, outputs=outputs, targets=y)

        self._stitcher.accumulate_val(outputs=outputs, xyxys=xyxys, dataloader_idx=dataloader_idx)
        return {"loss": per_sample_loss.mean()}

    def on_validation_epoch_end(self):
        sync_validation_accumulators(self)
        log_validation_epoch_metrics(self)
        self._stitcher.on_validation_epoch_end(self)

    def configure_optimizers(self):
        return configure_optimizers_runtime(self)


__all__ = ["RegressionPLModel"]
