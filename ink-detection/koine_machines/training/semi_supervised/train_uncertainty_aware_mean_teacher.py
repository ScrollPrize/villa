import json
import math
import os
from copy import deepcopy

import accelerate
import click
import torch
import wandb
from accelerate.utils import (
    DistributedDataParallelKwargs,
    GradientAccumulationPlugin,
    set_seed,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

from koine_machines.data.ink_dataset import InkDataset
from koine_machines.evaluation.metrics.balanced_accuracy import BalancedAccuracy
from koine_machines.evaluation.metrics.confusion import Confusion, ConfusionCounts
from koine_machines.models.load_checkpoint import (
    load_training_checkpoint_from_config,
    restore_training_state,
)
from koine_machines.models.make_model import make_model
from koine_machines.training.deep_supervision import build_deep_supervision_targets
from koine_machines.training.loss.losses import create_loss_from_config
from koine_machines.training.semi_supervised.ramps import sigmoid_rampup
from koine_machines.training.semi_supervised.two_stream_batch_sampler import (
    TwoStreamBatchSampler,
)
from koine_machines.training.stitching import resolve_model_and_loader_patch_sizes
from koine_machines.training.train import ValidationMetricBatch, _forward_model
from vesuvius.models.training.lr_schedulers import get_scheduler
from vesuvius.models.training.optimizers import (
    OptimizerParamGroupTarget,
    create_optimizer,
)
from vesuvius.models.utils import InitWeights_He


def _ssl_config_value(config, key, default):
    ssl_config = config.get("semi_supervised") or {}
    return ssl_config.get(key, config.get(key, default))


def _prepare_projected_loss_inputs(preds, batch):
    if isinstance(preds, (list, tuple)):
        loss_preds = []
        targets = None
        ignore_mask = None
        for idx, pred_level in enumerate(preds):
            current_preds, current_targets, current_ignore_mask = _prepare_projected_loss_inputs(
                pred_level,
                batch,
            )
            loss_preds.append(current_preds)
            if idx == 0:
                targets = current_targets
                ignore_mask = current_ignore_mask
        return type(preds)(loss_preds), targets, ignore_mask

    targets = (torch.amax(batch["inklabels"], dim=2) > 0).to(dtype=batch["inklabels"].dtype)
    supervision_mask = torch.amax(batch["supervision_mask"], dim=2)
    ignore_mask = (supervision_mask <= 0).to(dtype=targets.dtype)
    return preds, targets, ignore_mask


def _select_batch_items(value, mask):
    if isinstance(value, (list, tuple)):
        return type(value)(_select_batch_items(item, mask) for item in value)
    return value[mask]


def _build_targets_with_ignore(targets, ignore_mask, loss_preds):
    targets_with_ignore = build_deep_supervision_targets(targets, loss_preds, mode="nearest")
    ds_ignore = build_deep_supervision_targets(ignore_mask, loss_preds, mode="nearest")
    if isinstance(targets_with_ignore, (list, tuple)):
        return type(targets_with_ignore)(
            torch.cat([target_level, ignore_level], dim=1)
            for target_level, ignore_level in zip(targets_with_ignore, ds_ignore)
        )
    return torch.cat([targets_with_ignore, ds_ignore], dim=1)


def _compute_supervised_loss(loss_fn, loss_preds, targets, ignore_mask):
    targets_with_ignore = _build_targets_with_ignore(targets, ignore_mask, loss_preds)
    return loss_fn(
        type(loss_preds)(v.float() for v in loss_preds) if isinstance(loss_preds, (list, tuple)) else loss_preds.float(),
        (
            type(targets_with_ignore)(v.float() for v in targets_with_ignore)
            if isinstance(targets_with_ignore, (list, tuple))
            else targets_with_ignore.float()
        ),
    )


def _get_noise_bound(inputs, noise_scale):
    input_std = float(inputs.detach().float().std().item()) if inputs.numel() > 0 else 0.0
    return 2.0 * float(noise_scale) * max(input_std, 0.1)


def _compute_teacher_probabilities(
    ema_model,
    unlabeled_inputs,
    *,
    model_crop_size,
    use_stitched_forward,
    stitched_gradient_checkpointing,
    uncertainty_T,
    noise_scale,
):
    if int(uncertainty_T) < 2:
        raise ValueError(f"uncertainty_T must be >= 2, got {uncertainty_T}")

    noise_bound = _get_noise_bound(unlabeled_inputs, noise_scale)
    all_probs = []
    with torch.no_grad():
        for _ in range(int(uncertainty_T)):
            noise = torch.zeros_like(unlabeled_inputs)
            if float(noise_scale) > 0:
                noise = torch.clamp(
                    torch.randn_like(unlabeled_inputs) * float(noise_scale),
                    -noise_bound,
                    noise_bound,
                )
            teacher_logits = _forward_model(
                ema_model,
                unlabeled_inputs + noise,
                model_crop_size,
                stitched=use_stitched_forward,
                use_gradient_checkpointing=stitched_gradient_checkpointing,
            )
            primary_teacher_logits = teacher_logits[0] if isinstance(teacher_logits, (list, tuple)) else teacher_logits
            all_probs.append(torch.sigmoid(primary_teacher_logits.float()))

    mean_prob = torch.stack(all_probs, dim=0).mean(dim=0)
    mean_prob = mean_prob.clamp_(1e-6, 1.0 - 1e-6)
    uncertainty = -(
        mean_prob * torch.log(mean_prob) + (1.0 - mean_prob) * torch.log(1.0 - mean_prob)
    )
    return mean_prob, uncertainty


def _get_consistency_weight(consistency_weight, step, rampup_steps):
    return float(consistency_weight) * sigmoid_rampup(float(step), float(rampup_steps))


def _get_uncertainty_threshold(start, end, step, max_steps):
    ramp = sigmoid_rampup(float(step), float(max_steps))
    return (float(start) + (float(end) - float(start)) * ramp) * math.log(2.0)


def _initialize_wandb(config, checkpoint):
    if "wandb_project" not in config:
        return
    wandb_kwargs = {
        "project": config["wandb_project"],
        "entity": config["wandb_entity"],
        "config": config,
    }
    if config.get("wandb_resume", False):
        wandb_run_id = config.get("wandb_run_id")
        if not wandb_run_id and checkpoint is not None:
            wandb_run_id = checkpoint.get("wandb_run_id")
        if not wandb_run_id:
            raise ValueError(
                "wandb_resume=true requires wandb_run_id in config or checkpoint"
            )
        wandb_kwargs["id"] = wandb_run_id
        wandb_kwargs["resume"] = "must"
    wandb.init(**wandb_kwargs)


def _build_train_dataloader(train_ds, config, accelerator):
    labeled_indices, unlabeled_indices = train_ds.get_labeled_unlabeled_patch_indices()
    if not labeled_indices:
        raise ValueError("Semi-supervised training requires at least one labeled patch")
    if not unlabeled_indices:
        raise ValueError("Semi-supervised training requires at least one unlabeled patch")

    labeled_batch_size = int(_ssl_config_value(config, "labeled_batch_size", max(1, int(config["batch_size"]) // 2)))
    if labeled_batch_size >= int(config["batch_size"]):
        raise ValueError("labeled_batch_size must be smaller than batch_size")

    selection_seed = int(config["seed"])
    labeled_order = list(labeled_indices)
    unlabeled_order = list(unlabeled_indices)
    torch_rng = torch.Generator().manual_seed(selection_seed)
    labeled_perm = torch.randperm(len(labeled_order), generator=torch_rng).tolist()
    unlabeled_perm = torch.randperm(len(unlabeled_order), generator=torch_rng).tolist()
    labeled_order = [labeled_order[idx] for idx in labeled_perm]
    unlabeled_order = [unlabeled_order[idx] for idx in unlabeled_perm]

    num_labeled = _ssl_config_value(config, "num_labeled", None)
    labeled_ratio = float(_ssl_config_value(config, "labeled_ratio", 1.0))
    if num_labeled is None:
        num_labeled = int(labeled_ratio * max(1, len(labeled_order)))
    num_labeled = max(int(num_labeled), labeled_batch_size)
    selected_labeled = labeled_order[: min(len(labeled_order), int(num_labeled))]

    unlabeled_batch_size = int(config["batch_size"]) - labeled_batch_size
    if len(unlabeled_order) < unlabeled_batch_size:
        raise ValueError(
            f"Need at least {unlabeled_batch_size} unlabeled patches, found {len(unlabeled_order)}"
        )

    dataloader_workers = int(config.get("dataloader_workers", 0))
    dataloader_kwargs = {
        "pin_memory": bool(config.get("pin_memory", accelerator.device.type == "cuda")),
        "num_workers": dataloader_workers,
    }
    if dataloader_workers > 0:
        dataloader_kwargs["multiprocessing_context"] = "spawn"
        dataloader_kwargs["persistent_workers"] = True

    batch_sampler = TwoStreamBatchSampler(
        primary_indices=selected_labeled,
        secondary_indices=unlabeled_order,
        batch_size=int(config["batch_size"]),
        secondary_batch_size=unlabeled_batch_size,
        seed=selection_seed,
    )

    train_dl = DataLoader(
        train_ds,
        batch_sampler=batch_sampler,
        **dataloader_kwargs,
    )
    return train_dl, selected_labeled, unlabeled_order


@click.command()
@click.argument("config_path", type=click.Path(exists=True))
def train(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    checkpoint_path, checkpoint, weights_only = load_training_checkpoint_from_config(config, config_path)
    resume_full_state = checkpoint is not None and not weights_only
    start_step = 0

    if str(config.get("model_type", "")).strip().lower() == "dinov2":
        model_config = config.setdefault("model_config", {})
        for key in ("pretrained_backbone", "pretrained_decoder_type"):
            if key in config:
                model_config.setdefault(key, config[key])
        if not model_config.get("pretrained_backbone"):
            raise ValueError(
                "model_type='dinov2' requires model_config.pretrained_backbone "
                "or a top-level pretrained_backbone entry"
            )
        config["model_type"] = "vesuvius_unet"

    mode = str(config.get("mode", "flat")).strip().lower()
    if mode in {"normal_pooled_3d", "full_3d"}:
        raise ValueError(
            "Semi-supervised uncertainty-aware mean teacher currently supports only the projected training path"
        )

    if not config.get("unlabeled_datasets"):
        raise ValueError("semi-supervised training requires a non-empty unlabeled_datasets config entry")

    config.setdefault("volume_auth_json", None)
    config["targets"]["ink"]["out_channels"] = 1
    config["targets"]["ink"]["activation"] = "none"
    config["in_channels"] = 1
    learning_rate = config.get("learning_rate", 0.01)
    grad_acc_steps = int(config.get("grad_acc_steps", 1))
    grad_clip = config.get("grad_clip")
    max_steps = config.get("max_steps", math.ceil(config["num_iterations"] / grad_acc_steps))
    val_every = int(config.get("val_every", 500))
    save_every = int(config.get("save_every", val_every))
    log_every = int(config.get("log_every", 1))

    requested_stitch_factor = int(config.get("stitch_factor", 1))
    use_stitched_forward = bool(config.get("use_stitched_forward", requested_stitch_factor > 1))
    stitched_gradient_checkpointing = bool(config.get("stitched_gradient_checkpointing", True))
    model_crop_size = tuple(config["patch_size"])
    loader_patch_size = model_crop_size
    stitch_factor = 1
    if use_stitched_forward:
        model_crop_size, loader_patch_size, stitch_factor = resolve_model_and_loader_patch_sizes(config)
    config["crop_size"] = list(model_crop_size)
    config["patch_size"] = list(model_crop_size)
    config["stitch_factor"] = stitch_factor
    config["use_stitched_forward"] = use_stitched_forward
    config["stitched_gradient_checkpointing"] = stitched_gradient_checkpointing

    dataloader_config = accelerate.DataLoaderConfiguration(non_blocking=True)
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=bool(config.get("ddp_find_unused_parameters", True)),
        broadcast_buffers=bool(config.get("ddp_broadcast_buffers", False)),
    )
    gradient_accumulation_plugin = GradientAccumulationPlugin(
        num_steps=grad_acc_steps,
        sync_with_dataloader=False,
    )
    accelerator = accelerate.Accelerator(
        mixed_precision=config.get("mixed_precision", "fp16"),
        gradient_accumulation_plugin=gradient_accumulation_plugin,
        dataloader_config=dataloader_config,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        _initialize_wandb(config, checkpoint)

    out_dir = config["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    set_seed(int(config["seed"]))

    dataset_config = deepcopy(config)
    dataset_config["patch_size"] = list(loader_patch_size)

    labeled_shared_ds = InkDataset(dataset_config, do_augmentations=False)
    if len(labeled_shared_ds.training_patches) == 0:
        raise ValueError("InkDataset produced no labeled training patches after applying supervision masking")

    unlabeled_dataset_config = deepcopy(dataset_config)
    unlabeled_dataset_config["patch_discovery_mode"] = "unlabeled"
    unlabeled_shared_ds = InkDataset(unlabeled_dataset_config, do_augmentations=False)
    if len(unlabeled_shared_ds.unlabeled_patches) == 0:
        raise ValueError("InkDataset produced no unlabeled training patches")

    train_patches = list(labeled_shared_ds.training_patches) + list(unlabeled_shared_ds.unlabeled_patches)
    train_ds = InkDataset(dataset_config, do_augmentations=True, patches=train_patches)
    val_ds = InkDataset(dataset_config, do_augmentations=False, patches=labeled_shared_ds.validation_patches)

    train_dl, labeled_indices, unlabeled_indices = _build_train_dataloader(train_ds, config, accelerator)

    dataloader_workers = int(config.get("dataloader_workers", 0))
    val_loader_kwargs = {
        "pin_memory": bool(config.get("pin_memory", accelerator.device.type == "cuda")),
        "num_workers": dataloader_workers,
    }
    if dataloader_workers > 0:
        val_loader_kwargs["multiprocessing_context"] = "spawn"
        val_loader_kwargs["persistent_workers"] = True
    val_dl = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=len(val_ds) > 0,
        generator=torch.Generator().manual_seed(int(config["seed"]) + 1),
        **val_loader_kwargs,
    )

    model = make_model(config)
    optimizer_target = model
    pretrained_backbone = (config.get("model_config") or {}).get("pretrained_backbone")
    freeze_encoder = False
    if pretrained_backbone:
        freeze_encoder = bool(config.get("freeze_encoder", False))
        encoder_lr_mult = float(config.get("encoder_lr_mult", 1.0))

        encoder_params = list(model.shared_encoder.parameters())
        if freeze_encoder:
            for param in encoder_params:
                param.requires_grad = False

        if freeze_encoder or encoder_lr_mult != 1.0:
            encoder_param_ids = {id(param) for param in encoder_params}
            other_params = [
                param for param in model.parameters()
                if param.requires_grad and id(param) not in encoder_param_ids
            ]
            optimizer_target = []
            if other_params:
                optimizer_target.append({"params": other_params})
            if not freeze_encoder and encoder_params:
                optimizer_target.append(
                    {"params": encoder_params, "lr": learning_rate * encoder_lr_mult}
                )
            if not optimizer_target:
                raise ValueError("No trainable parameters remain after applying freeze_encoder")

    optimizer = create_optimizer(
        {
            "name": config.get("optimizer", "sgd"),
            "learning_rate": learning_rate,
            "weight_decay": config.get("weight_decay", 3e-5),
        },
        OptimizerParamGroupTarget(optimizer_target) if isinstance(optimizer_target, list) else optimizer_target,
    )
    lr_scheduler = get_scheduler(
        "diffusers_cosine_warmup",
        optimizer,
        initial_lr=learning_rate,
        max_steps=max_steps,
        warmup_steps=config.get("warmup_steps", 1000),
    )

    if not pretrained_backbone:
        model.apply(InitWeights_He(neg_slope=0.2))

    loss = create_loss_from_config(config)
    model, optimizer, train_dl, val_dl, lr_scheduler = accelerator.prepare(
        model,
        optimizer,
        train_dl,
        val_dl,
        lr_scheduler,
    )
    unwrapped_model = accelerator.unwrap_model(model)
    frozen_encoder = unwrapped_model.shared_encoder if pretrained_backbone and freeze_encoder else None
    ema_decay = float(_ssl_config_value(config, "ema_decay", 0.99))
    ema_model = deepcopy(unwrapped_model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer_step = 0
    if checkpoint is not None:
        start_step, optimizer_step = restore_training_state(
            unwrapped_model,
            optimizer,
            lr_scheduler,
            checkpoint,
            checkpoint_path,
            load_weights_only=weights_only,
            ema_model=ema_model,
        )
        accelerator.print(
            f"Loaded checkpoint '{checkpoint_path}'"
            + (f" and resuming from step {start_step}" if resume_full_state else " (weights only)")
        )

    uncertainty_T = int(_ssl_config_value(config, "uncertainty_T", 4))
    noise_scale = float(_ssl_config_value(config, "noise_scale", 0.1))
    consistency_weight = float(_ssl_config_value(config, "consistency_weight", 0.1))
    consistency_rampup = float(_ssl_config_value(config, "consistency_rampup", max(1, max_steps // 10)))
    uncertainty_threshold_start = float(_ssl_config_value(config, "uncertainty_threshold_start", 0.75))
    uncertainty_threshold_end = float(_ssl_config_value(config, "uncertainty_threshold_end", 1.0))

    latest_val_loss = None
    latest_ema_val_loss = None
    validation_confusion_metric = Confusion()
    validation_balanced_accuracy_metric = BalancedAccuracy()

    train_iterator = iter(train_dl)
    progress_bar = tqdm(
        range(start_step, int(config["num_iterations"])),
        total=int(config["num_iterations"]),
        initial=start_step,
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    def get_model_input(batch):
        return batch["image"].float()

    def refresh_progress_bar(current_train_loss):
        if not accelerator.is_main_process:
            return
        postfix = {"loss": f"{current_train_loss:.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"}
        if latest_val_loss is not None:
            postfix["val_loss"] = f"{latest_val_loss:.4f}"
        if latest_ema_val_loss is not None:
            postfix["ema_val_loss"] = f"{latest_ema_val_loss:.4f}"
        progress_bar.set_postfix(postfix, refresh=False)
        progress_bar.update(0)

    accelerator.print(
        f"Semi-supervised split: {len(labeled_indices)} labeled patches, {len(unlabeled_indices)} unlabeled patches"
    )

    for step in progress_bar:
        model.train()
        if frozen_encoder is not None:
            frozen_encoder.eval()

        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dl)
            batch = next(train_iterator)

        with accelerator.accumulate(model):
            model_inputs = get_model_input(batch)
            is_unlabeled = batch["is_unlabeled"].to(device=accelerator.device).bool()
            labeled_mask = ~is_unlabeled
            unlabeled_mask = is_unlabeled
            if not labeled_mask.any():
                raise ValueError("Semi-supervised batch did not contain labeled samples")
            if not unlabeled_mask.any():
                raise ValueError("Semi-supervised batch did not contain unlabeled samples")

            with accelerator.autocast():
                preds = _forward_model(
                    model,
                    model_inputs,
                    model_crop_size,
                    stitched=use_stitched_forward,
                    use_gradient_checkpointing=stitched_gradient_checkpointing,
                )
                loss_preds, targets, ignore_mask = _prepare_projected_loss_inputs(preds, batch)
                labeled_loss_preds = _select_batch_items(loss_preds, labeled_mask)
                labeled_targets = targets[labeled_mask]
                labeled_ignore_mask = ignore_mask[labeled_mask]
                supervised_loss = _compute_supervised_loss(
                    loss,
                    labeled_loss_preds,
                    labeled_targets,
                    labeled_ignore_mask,
                )

            primary_loss_preds = loss_preds[0] if isinstance(loss_preds, (list, tuple)) else loss_preds
            unlabeled_inputs = model_inputs[unlabeled_mask]
            student_unlabeled_logits = primary_loss_preds[unlabeled_mask]
            teacher_mean_prob, uncertainty = _compute_teacher_probabilities(
                ema_model,
                unlabeled_inputs,
                model_crop_size=model_crop_size,
                use_stitched_forward=use_stitched_forward,
                stitched_gradient_checkpointing=stitched_gradient_checkpointing,
                uncertainty_T=uncertainty_T,
                noise_scale=noise_scale,
            )
            consistency_dist = (torch.sigmoid(student_unlabeled_logits.float()) - teacher_mean_prob).pow(2)
            uncertainty_threshold = _get_uncertainty_threshold(
                uncertainty_threshold_start,
                uncertainty_threshold_end,
                optimizer_step,
                max_steps,
            )
            consistency_mask = (uncertainty < uncertainty_threshold).to(dtype=consistency_dist.dtype)
            mask_normalizer = consistency_mask.expand_as(consistency_dist).sum().clamp_min(1.0)
            consistency_loss = (
                consistency_dist * consistency_mask.expand_as(consistency_dist)
            ).sum() / mask_normalizer
            current_consistency_weight = _get_consistency_weight(
                consistency_weight,
                optimizer_step,
                consistency_rampup,
            )
            total_loss = supervised_loss + (current_consistency_weight * consistency_loss)

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss at step {step}")

            accelerator.backward(total_loss)
            if grad_clip is not None and grad_clip > 0 and accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                optimizer_step += 1
                ema_state = ema_model.state_dict()
                for name, model_value in unwrapped_model.state_dict().items():
                    ema_value = ema_state[name]
                    model_value = model_value.detach()
                    if torch.is_floating_point(ema_value):
                        ema_value.lerp_(model_value.to(dtype=ema_value.dtype), 1.0 - ema_decay)
                    else:
                        ema_value.copy_(model_value)

        train_loss = float(
            accelerator.reduce(
                total_loss.detach().to(device=accelerator.device, dtype=torch.float32).reshape(()),
                reduction="mean",
            ).item()
        )
        supervised_loss_value = float(
            accelerator.reduce(
                supervised_loss.detach().to(device=accelerator.device, dtype=torch.float32).reshape(()),
                reduction="mean",
            ).item()
        )
        consistency_loss_value = float(
            accelerator.reduce(
                consistency_loss.detach().to(device=accelerator.device, dtype=torch.float32).reshape(()),
                reduction="mean",
            ).item()
        )
        mask_fraction = float(
            accelerator.reduce(
                consistency_mask.float().mean().detach().to(device=accelerator.device, dtype=torch.float32).reshape(()),
                reduction="mean",
            ).item()
        )

        if accelerator.is_main_process:
            refresh_progress_bar(train_loss)

        if accelerator.is_main_process and step % log_every == 0:
            log_dict = {
                "train/loss": train_loss,
                "train/supervised_loss": supervised_loss_value,
                "train/consistency_loss": consistency_loss_value,
                "train/consistency_weight": current_consistency_weight,
                "train/uncertainty_mask_fraction": mask_fraction,
                "train/lr": optimizer.param_groups[0]["lr"],
                "step": step,
            }
            latest_loss_metrics = getattr(loss, "latest_metrics", None)
            if isinstance(latest_loss_metrics, dict):
                for name, metric_value in latest_loss_metrics.items():
                    metric_tensor = (
                        metric_value.detach()
                        if isinstance(metric_value, torch.Tensor)
                        else torch.tensor(float(metric_value), device=accelerator.device)
                    )
                    log_dict[str(name)] = float(
                        accelerator.reduce(
                            metric_tensor.to(device=accelerator.device, dtype=torch.float32).reshape(()),
                            reduction="mean",
                        ).item()
                    )
            if wandb.run is not None:
                wandb.log(log_dict, step=step)

        if step % val_every == 0 and step > 0:
            model.eval()
            val_loss_total = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            val_loss_batches = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            ema_val_loss_total = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            ema_val_loss_batches = torch.zeros((), device=accelerator.device, dtype=torch.float32)
            validation_counts = Confusion.zero_counts(device=accelerator.device)

            num_val_batches = min(len(val_dl), int(config.get("val_steps", 10)))
            if num_val_batches == 0:
                continue
            val_iterator = iter(val_dl)
            with torch.no_grad():
                for _ in range(num_val_batches):
                    val_batch = next(val_iterator)
                    val_inputs = get_model_input(val_batch)
                    with accelerator.autocast():
                        val_preds = _forward_model(
                            model,
                            val_inputs,
                            model_crop_size,
                            stitched=use_stitched_forward,
                            use_gradient_checkpointing=stitched_gradient_checkpointing,
                        )
                        val_loss_preds, val_targets, val_ignore_mask = _prepare_projected_loss_inputs(val_preds, val_batch)
                        val_l = _compute_supervised_loss(loss, val_loss_preds, val_targets, val_ignore_mask)
                    primary_val_loss_preds = val_loss_preds[0] if isinstance(val_loss_preds, (list, tuple)) else val_loss_preds
                    val_loss_total = val_loss_total + accelerator.reduce(
                        val_l.detach().to(device=accelerator.device, dtype=torch.float32).reshape(()),
                        reduction="mean",
                    )
                    val_loss_batches = val_loss_batches + 1.0
                    batch_counts = validation_confusion_metric.compute_batch(
                        ValidationMetricBatch(
                            logits=primary_val_loss_preds.detach(),
                            targets=val_targets.detach(),
                            valid_mask=(val_ignore_mask <= 0).detach(),
                        )
                    )
                    gathered_batch_counts = accelerator.gather_for_metrics(
                        torch.stack((batch_counts.tp, batch_counts.fp, batch_counts.fn, batch_counts.tn)).unsqueeze(0)
                    )
                    validation_counts = Confusion.add_counts(
                        validation_counts,
                        ConfusionCounts(
                            tp=gathered_batch_counts[:, 0].sum(),
                            fp=gathered_batch_counts[:, 1].sum(),
                            fn=gathered_batch_counts[:, 2].sum(),
                            tn=gathered_batch_counts[:, 3].sum(),
                        ),
                    )

                    with accelerator.autocast():
                        ema_val_preds = _forward_model(
                            ema_model,
                            val_inputs,
                            model_crop_size,
                            stitched=use_stitched_forward,
                            use_gradient_checkpointing=stitched_gradient_checkpointing,
                        )
                        ema_val_loss_preds, _, _ = _prepare_projected_loss_inputs(ema_val_preds, val_batch)
                        ema_val_l = _compute_supervised_loss(loss, ema_val_loss_preds, val_targets, val_ignore_mask)
                    ema_val_loss_total = ema_val_loss_total + accelerator.reduce(
                        ema_val_l.detach().to(device=accelerator.device, dtype=torch.float32).reshape(()),
                        reduction="mean",
                    )
                    ema_val_loss_batches = ema_val_loss_batches + 1.0

            mean_val_loss = float((val_loss_total / val_loss_batches.clamp_min(1.0)).item())
            mean_ema_val_loss = float((ema_val_loss_total / ema_val_loss_batches.clamp_min(1.0)).item())
            if accelerator.is_main_process:
                latest_val_loss = mean_val_loss
                latest_ema_val_loss = mean_ema_val_loss
                refresh_progress_bar(train_loss)
                balanced_accuracy = float(
                    validation_balanced_accuracy_metric._from_counts(validation_counts).item()
                )
                if wandb.run is not None:
                    wandb.log(
                        {
                            "val/loss": mean_val_loss,
                            "val/ema_loss": mean_ema_val_loss,
                            "val/balanced_accuracy": balanced_accuracy,
                            "val/tp": float(validation_counts.tp.item()),
                            "val/fp": float(validation_counts.fp.item()),
                            "val/fn": float(validation_counts.fn.item()),
                            "val/tn": float(validation_counts.tn.item()),
                            "step": step,
                        },
                        step=step,
                    )

        if accelerator.is_main_process and step % save_every == 0 and step > 0:
            checkpoint_to_save = {
                "model": accelerator.get_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "ema_model": ema_model.state_dict(),
                "ema_optimizer_step": optimizer_step,
                "config": config,
                "step": step,
                "wandb_run_id": wandb.run.id if wandb.run is not None else config.get("wandb_run_id"),
            }
            torch.save(checkpoint_to_save, f"{out_dir}/ckpt_{step:06}.pth")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
