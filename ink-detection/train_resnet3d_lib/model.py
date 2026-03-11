import os.path as osp

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
from train_resnet3d_lib.stitch_manager import StitchManager


_RESNET3D_ENCODER_DIMS = {
    50: [256, 512, 1024, 2048],
    101: [256, 512, 1024, 2048],
    152: [256, 512, 1024, 2048],
}


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


def _init_group_dro_if_needed(model, *, group_counts):
    model.group_dro = None
    if model.objective != "group_dro":
        return

    model.group_dro = GroupDROComputer(
        n_groups=model.n_groups,
        group_counts=group_counts,
        alpha=getattr(CFG, "group_dro_alpha", None),
        gamma=float(getattr(CFG, "group_dro_gamma", 0.1)),
        adj=getattr(CFG, "group_dro_adj", None),
        min_var_weight=float(getattr(CFG, "group_dro_min_var_weight", 0.0)),
        step_size=getattr(CFG, "robust_step_size", None),
        normalize_loss=bool(getattr(CFG, "group_dro_normalize_loss", False)),
        btl=bool(getattr(CFG, "group_dro_btl", False)),
    )


def _resolve_resnet3d_depth():
    resnet3d_model_depth = int(getattr(CFG, "resnet3d_model_depth", None))
    if resnet3d_model_depth not in _RESNET3D_ENCODER_DIMS:
        raise ValueError(
            f"Unsupported resnet3d_model_depth={resnet3d_model_depth!r}. "
            f"Expected one of {sorted(_RESNET3D_ENCODER_DIMS)!r}."
        )
    return resnet3d_model_depth


def _build_backbone_with_optional_pretrained(*, resnet3d_model_depth):
    backbone = generate_model(
        model_depth=resnet3d_model_depth,
        n_input_channels=1,
        forward_features=True,
        n_classes=1039,
    )

    norm = str(getattr(CFG, "norm", "batch")).lower()
    group_norm_groups = int(getattr(CFG, "group_norm_groups", 32))
    init_ckpt_path = getattr(CFG, "init_ckpt_path", None)
    use_pretrained = bool(getattr(CFG, "pretrained", True))
    if use_pretrained and (not init_ckpt_path):
        backbone_pretrained_path = str(getattr(CFG, "backbone_pretrained_path", "")).strip()
        if not osp.exists(backbone_pretrained_path):
            raise FileNotFoundError(
                f"Missing backbone pretrained weights: {backbone_pretrained_path}. "
                "Set model.backbone_pretrained_path to a valid file, "
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


def _encoder_dims_for_resnet3d_depth(model_depth):
    return _RESNET3D_ENCODER_DIMS[model_depth]


def initialize_regression_state(
    model,
    *,
    total_steps,
    group_names,
    stitch_group_idx_by_segment,
    group_counts,
):
    model.objective = str(getattr(CFG, "objective", "erm")).lower()
    model.loss_mode = str(getattr(CFG, "loss_mode", "batch")).lower()
    model.loss_recipe = str(getattr(CFG, "loss_recipe", "dice_bce")).lower()
    model.bce_smooth_factor = float(getattr(CFG, "bce_smooth_factor", 0.25))
    model.soft_label_positive = float(getattr(CFG, "soft_label_positive", 1.0))
    model.soft_label_negative = float(getattr(CFG, "soft_label_negative", 0.0))
    model.with_norm = False
    model.total_steps = int(total_steps)

    model.n_groups = int(len(group_names))
    model.group_names = list(group_names)
    model._stitch_group_idx_by_segment = _normalize_stitch_group_idx_by_segment(
        stitch_group_idx_by_segment,
        n_groups=model.n_groups,
    )

    _init_group_dro_if_needed(model, group_counts=group_counts)

    model.erm_group_topk = int(getattr(CFG, "erm_group_topk", 0) or 0)

    model._ema_decay = float(getattr(CFG, "ema_decay", 0.9))
    model._ema_metrics = {}

    model._eval_threshold = float(getattr(CFG, "eval_threshold", 0.5))
    model._val_eval_metrics = None

    model.loss_func1 = smp.losses.DiceLoss(mode="binary")

    resnet3d_model_depth = _resolve_resnet3d_depth()
    model.backbone = _build_backbone_with_optional_pretrained(
        resnet3d_model_depth=resnet3d_model_depth,
    )

    norm = str(getattr(CFG, "norm", "batch")).lower()
    group_norm_groups = int(getattr(CFG, "group_norm_groups", 32))
    encoder_dims = _encoder_dims_for_resnet3d_depth(resnet3d_model_depth)

    model.decoder = Decoder(encoder_dims=encoder_dims, upscale=1, norm=norm, group_norm_groups=group_norm_groups)

    if model.with_norm:
        if norm == "group":
            model.normalization = nn.GroupNorm(num_groups=1, num_channels=1)
        else:
            model.normalization = nn.BatchNorm3d(num_features=1)


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

    if loss_recipe not in {"dice_bce", "bce_only"}:
        raise ValueError(f"Unknown training.loss_recipe: {loss_recipe!r}")

    if objective == "group_dro":
        if loss_mode != "per_sample":
            raise ValueError("training.objective=group_dro requires training.loss_mode=per_sample")
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

    if objective != "erm":
        raise ValueError(f"Unknown training.objective: {objective!r}")

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

    if loss_mode != "per_sample":
        raise ValueError(f"Unknown training.loss_mode: {loss_mode!r}")

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
        total_steps=0,
        group_names=None,
        stitch_group_idx_by_segment=None,
        group_counts=None,
        stitch_manager: StitchManager | None = None,
    ):
        super(RegressionPLModel, self).__init__()
        group_names = [] if group_names is None else list(group_names)
        stitch_group_idx_by_segment = {} if stitch_group_idx_by_segment is None else dict(stitch_group_idx_by_segment)
        group_counts_hparams = list(group_counts) if group_counts is not None else None
        self._stitcher = stitch_manager if stitch_manager is not None else StitchManager()
        self.save_hyperparameters(
            {
                "total_steps": int(total_steps),
                "group_names": group_names,
                "stitch_group_idx_by_segment": stitch_group_idx_by_segment,
                "group_counts": group_counts_hparams,
            }
        )
        initialize_regression_state(
            self,
            total_steps=total_steps,
            group_names=group_names,
            stitch_group_idx_by_segment=stitch_group_idx_by_segment,
            group_counts=group_counts,
        )

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
