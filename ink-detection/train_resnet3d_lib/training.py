import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb

from train_resnet3d_lib.config import (
    CFG,
    log,
)
from train_resnet3d_lib.model import RegressionPLModel
from train_resnet3d_lib.runtime.checkpointing import load_state_dict_from_checkpoint
from train_resnet3d_lib.runtime.wandb_local_metrics import LocalMetricsWandbLogger


def _resolve_trainer_precision(cli_precision):
    if cli_precision is not None:
        cli_precision_text = str(cli_precision).strip().lower()
        if cli_precision_text and cli_precision_text != "auto":
            return cli_precision
    use_amp = bool(getattr(CFG, "use_amp", True))
    return "16-mixed" if use_amp else "32-true"


def _build_regression_model_state(run_state, data_state):
    # Keep one transport payload for model construction, but build it in
    # conceptual sections so research edits stay easy to locate.
    model_state = {
        "with_norm": False,
        "total_steps": int(data_state["steps_per_epoch"]),
        "n_groups": int(len(data_state["group_names"])),
        "group_names": list(data_state["group_names"]),
        "stitch_group_idx_by_segment": dict(data_state["group_idx_by_segment"]),
        "norm": str(getattr(CFG, "norm", "batch")),
        "group_norm_groups": int(getattr(CFG, "group_norm_groups", 32)),
    }

    objective_state = {
        "objective": str(getattr(CFG, "objective", "erm")),
        "loss_mode": str(getattr(CFG, "loss_mode", "batch")),
        "loss_recipe": str(getattr(CFG, "loss_recipe", "dice_bce")).lower(),
        "bce_smooth_factor": float(getattr(CFG, "bce_smooth_factor", 0.25)),
        "soft_label_positive": float(getattr(CFG, "soft_label_positive", 1.0)),
        "soft_label_negative": float(getattr(CFG, "soft_label_negative", 0.0)),
        "robust_step_size": run_state["robust_step_size"],
        "group_counts": list(data_state["train_group_counts"]),
        "group_dro_gamma": float(run_state["group_dro_gamma"]),
        "group_dro_btl": bool(run_state["group_dro_btl"]),
        "group_dro_alpha": run_state["group_dro_alpha"],
        "group_dro_normalize_loss": bool(run_state["group_dro_normalize_loss"]),
        "group_dro_min_var_weight": float(run_state["group_dro_min_var_weight"]),
        "group_dro_adj": run_state["group_dro_adj"],
        "erm_group_topk": int(getattr(CFG, "erm_group_topk", 0)),
    }

    stitch_state = {
        "stitch_val_dataloader_idx": data_state["stitch_val_dataloader_idx"],
        "stitch_pred_shape": data_state["stitch_pred_shape"],
        "stitch_segment_id": (
            None if data_state["stitch_segment_id"] is None else str(data_state["stitch_segment_id"])
        ),
        "stitch_all_val": bool(getattr(CFG, "stitch_all_val", False)),
        "stitch_downsample": int(getattr(CFG, "stitch_downsample", 1)),
        "stitch_all_val_shapes": list(data_state["val_stitch_shapes"]),
        "stitch_all_val_segment_ids": [str(x) for x in data_state["val_stitch_segment_ids"]],
        "stitch_train_shapes": list(data_state["train_stitch_shapes"]),
        "stitch_train_segment_ids": [str(x) for x in data_state["train_stitch_segment_ids"]],
        "stitch_use_roi": bool(getattr(CFG, "stitch_use_roi", False)),
        "stitch_val_bboxes": dict(data_state.get("val_mask_bboxes") or {}),
        "stitch_train_bboxes": dict(data_state.get("train_mask_bboxes") or {}),
        "stitch_log_only_shapes": list(data_state.get("log_only_stitch_shapes") or []),
        "stitch_log_only_segment_ids": [str(x) for x in (data_state.get("log_only_stitch_segment_ids") or [])],
        "stitch_log_only_bboxes": dict(data_state.get("log_only_mask_bboxes") or {}),
        "stitch_log_only_downsample": int(
            getattr(CFG, "stitch_log_only_downsample", getattr(CFG, "stitch_downsample", 1))
        ),
        "stitch_log_only_every_n_epochs": int(getattr(CFG, "stitch_log_only_every_n_epochs", 10)),
        "stitch_train": bool(getattr(CFG, "stitch_train", False)),
        "stitch_train_every_n_epochs": int(getattr(CFG, "stitch_train_every_n_epochs", 1)),
    }
    model_state.update(objective_state)
    model_state.update(stitch_state)
    return model_state


def build_model(run_state, data_state, wandb_logger):
    model_state = _build_regression_model_state(run_state, data_state)
    model = RegressionPLModel(
        model_state=model_state,
    )
    if data_state["train_stitch_loaders"]:
        model.set_train_stitch_loaders(data_state["train_stitch_loaders"], data_state["train_stitch_segment_ids"])
    if data_state.get("log_only_stitch_loaders"):
        model.set_log_only_stitch_loaders(
            data_state.get("log_only_stitch_loaders"),
            data_state.get("log_only_stitch_segment_ids"),
        )
    if data_state["include_train_xyxys"]:
        model.set_stitch_borders(
            train_borders=data_state["train_mask_borders"],
            val_borders=data_state["val_mask_borders"],
        )
    if run_state["init_ckpt_path"]:
        if bool(getattr(CFG, "pretrained", True)):
            log(f"loading init weights from {run_state['init_ckpt_path']}")
            state_dict = load_state_dict_from_checkpoint(run_state["init_ckpt_path"])
            model.load_state_dict(state_dict, strict=True)
            log("loaded init weights (strict=True)")
        else:
            log("CFG.pretrained=False; skipped init_ckpt_path weight loading.")
    if wandb_logger is not None:
        wandb_logger.watch(model, log="all", log_freq=100)
    return model


def build_trainer(args, wandb_logger):
    trainer_logger = wandb_logger if wandb_logger is not None else False
    precision = _resolve_trainer_precision(getattr(args, "precision", None))
    log(f"trainer precision={precision!r} (cli={getattr(args, 'precision', None)!r} cfg.use_amp={bool(getattr(CFG, 'use_amp', True))})")
    return pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=10,
        logger=trainer_logger,
        default_root_dir=CFG.outputs_path,
        accumulate_grad_batches=CFG.accumulate_grad_batches,
        precision=precision,
        gradient_clip_val=CFG.max_grad_norm,
        gradient_clip_algorithm="norm",
        callbacks=(
            [
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(
                    filename="best-epoch{epoch}",
                    dirpath=CFG.model_dir,
                    monitor="val/worst_group_loss",
                    mode="min",
                    save_top_k=1,
                    save_last=True,
                ),
            ]
            + (
                [
                    ModelCheckpoint(
                        filename="epoch{epoch}",
                        dirpath=CFG.model_dir,
                        every_n_epochs=int(getattr(CFG, "save_every_n_epochs", 1)),
                        save_top_k=-1,
                    )
                ]
                if CFG.save_every_epoch
                else []
            )
        ),
    )


def finalize_wandb_logging(trainer):
    logger = trainer.logger
    if logger is False or logger is None:
        return
    if not isinstance(logger, LocalMetricsWandbLogger):
        raise TypeError(
            "trainer.logger must be LocalMetricsWandbLogger when W&B logging is active, "
            f"got {type(logger).__name__}"
        )
    logger.persist_local_state()
    if wandb.run is not None:
        wandb.finish()


def fit(trainer, model, data_state, run_state):
    log("starting trainer.fit")
    trainer.fit(
        model=model,
        train_dataloaders=data_state["train_loader"],
        val_dataloaders=data_state["val_loaders"],
        ckpt_path=run_state["resume_ckpt_path"],
    )
    finalize_wandb_logging(trainer)


def validate(trainer, model, data_state, run_state):
    log("starting trainer.validate (stitch_only)")
    trainer.validate(
        model=model,
        dataloaders=data_state["val_loaders"],
        ckpt_path=run_state["resume_ckpt_path"],
        verbose=False,
    )
    finalize_wandb_logging(trainer)
