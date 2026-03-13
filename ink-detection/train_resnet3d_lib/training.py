import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from train_resnet3d_lib.config import (
    CFG,
    log,
)
from train_resnet3d_lib.model import RegressionPLModel
from train_resnet3d_lib.runtime.checkpointing import load_state_dict_from_checkpoint
from train_resnet3d_lib.runtime import wandb_runtime
from train_resnet3d_lib.stitch_manager import StitchManager


def _resolve_trainer_precision(cli_precision):
    if cli_precision is not None:
        cli_precision_text = str(cli_precision).strip().lower()
        if cli_precision_text and cli_precision_text != "auto":
            return cli_precision
    use_amp = bool(getattr(CFG, "use_amp", True))
    return "16-mixed" if use_amp else "32-true"


def build_model(run_state, data_state, wandb_logger):
    stitch_manager = StitchManager(
        stitch_val_dataloader_idx=data_state["stitch_val_dataloader_idx"],
        stitch_pred_shape=data_state["stitch_pred_shape"],
        stitch_segment_id=data_state["stitch_segment_id"],
        stitch_all_val_shapes=list(data_state["val_stitch_shapes"]),
        stitch_all_val_segment_ids=[str(x) for x in data_state["val_stitch_segment_ids"]],
        stitch_train_shapes=list(data_state["train_stitch_shapes"]),
        stitch_train_segment_ids=[str(x) for x in data_state["train_stitch_segment_ids"]],
        train_component_shapes=list(data_state.get("train_component_shapes") or []),
        train_component_keys=list(data_state.get("train_component_keys") or []),
        stitch_val_bboxes=dict(data_state.get("val_mask_bboxes") or {}),
        stitch_train_bboxes=dict(data_state.get("train_mask_bboxes") or {}),
        train_component_bboxes=dict(data_state.get("train_component_bboxes") or {}),
        stitch_log_only_shapes=list(data_state.get("log_only_stitch_shapes") or []),
        stitch_log_only_segment_ids=[str(x) for x in (data_state.get("log_only_stitch_segment_ids") or [])],
        stitch_log_only_bboxes=dict(data_state.get("log_only_mask_bboxes") or {}),
        train_loaders=list(data_state["train_stitch_loaders"]),
        train_component_datasets=list(data_state.get("train_component_datasets") or []),
        log_only_loaders=list(data_state.get("log_only_stitch_loaders") or []),
        train_borders=dict(data_state["train_mask_borders"] if data_state["include_train_xyxys"] else {}),
        val_borders=dict(data_state["val_mask_borders"] if data_state["include_train_xyxys"] else {}),
    )
    model = RegressionPLModel(
        total_steps=int(data_state["steps_per_epoch"]),
        group_names=list(data_state["group_names"]),
        stitch_group_idx_by_segment=dict(data_state["group_idx_by_segment"]),
        group_counts=list(data_state["train_group_counts"]),
        stitch_manager=stitch_manager,
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


def fit(trainer, model, data_state, run_state):
    log("starting trainer.fit")
    trainer.fit(
        model=model,
        train_dataloaders=data_state["train_loader"],
        val_dataloaders=data_state["val_loaders"],
        ckpt_path=run_state["resume_ckpt_path"],
    )
    wandb_runtime.finalize_wandb_logging(trainer)


def validate(trainer, model, data_state, run_state):
    log("starting trainer.validate (stitch_only)")
    trainer.validate(
        model=model,
        dataloaders=data_state["val_loaders"],
        ckpt_path=run_state["resume_ckpt_path"],
        verbose=False,
    )
    wandb_runtime.finalize_wandb_logging(trainer)
