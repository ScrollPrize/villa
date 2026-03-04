import torch
import torch.nn.functional as F

from metrics import StreamingBinarySegmentationMetrics
from train_resnet3d_lib.config import CFG


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

    if objective == "erm":
        if loss_mode == "batch":
            bce_targets = model.build_bce_targets(targets)
            bce_loss = F.binary_cross_entropy_with_logits(outputs, bce_targets)
            if loss_recipe == "dice_bce":
                dice_loss = model.loss_func1(outputs, targets)
                loss = 0.5 * dice_loss + 0.5 * bce_loss
            elif loss_recipe == "bce_only":
                dice_loss = per_sample_dice_loss.mean()
                loss = bce_loss
            else:
                raise ValueError(f"Unknown training.loss_recipe: {model.loss_recipe!r}")
            model.log("train/dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=False)
            model.log("train/bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=False)
            return loss

        if loss_mode != "per_sample":
            raise ValueError(f"Unknown training.loss_mode: {model.loss_mode!r}")

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

    if objective == "group_dro":
        if loss_mode != "per_sample":
            raise ValueError("GroupDRO requires training.loss_mode=per_sample")
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

    raise ValueError(f"Unknown training.objective: {model.objective!r}")


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


