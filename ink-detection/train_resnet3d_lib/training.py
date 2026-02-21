import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb

from train_resnet3d_lib.config import (
    CFG,
    log,
)
from train_resnet3d_lib.model import RegressionPLModel
from train_resnet3d_lib.checkpointing import load_state_dict_from_checkpoint


def build_model(run_state, data_state, wandb_logger):
    model = RegressionPLModel(
        enc='i3d',
        size=CFG.size,
        norm=getattr(CFG, "norm", "batch"),
        group_norm_groups=int(getattr(CFG, "group_norm_groups", 32)),
        objective=CFG.objective,
        loss_mode=CFG.loss_mode,
        erm_group_topk=int(getattr(CFG, "erm_group_topk", 0)),
        robust_step_size=run_state["robust_step_size"],
        group_counts=data_state["train_group_counts"],
        group_dro_gamma=run_state["group_dro_gamma"],
        group_dro_btl=run_state["group_dro_btl"],
        group_dro_alpha=run_state["group_dro_alpha"],
        group_dro_normalize_loss=run_state["group_dro_normalize_loss"],
        group_dro_min_var_weight=run_state["group_dro_min_var_weight"],
        group_dro_adj=run_state["group_dro_adj"],
        total_steps=data_state["steps_per_epoch"],
        n_groups=len(data_state["group_names"]),
        group_names=data_state["group_names"],
        stitch_group_idx_by_segment=data_state["group_idx_by_segment"],
        stitch_val_dataloader_idx=data_state["stitch_val_dataloader_idx"],
        stitch_pred_shape=data_state["stitch_pred_shape"],
        stitch_segment_id=data_state["stitch_segment_id"],
        stitch_all_val=bool(getattr(CFG, "stitch_all_val", False)),
        stitch_downsample=int(getattr(CFG, "stitch_downsample", 1)),
        stitch_all_val_shapes=data_state["val_stitch_shapes"],
        stitch_all_val_segment_ids=data_state["val_stitch_segment_ids"],
        stitch_train_shapes=data_state["train_stitch_shapes"],
        stitch_train_segment_ids=data_state["train_stitch_segment_ids"],
        stitch_use_roi=bool(getattr(CFG, "stitch_use_roi", False)),
        stitch_val_bboxes=data_state.get("val_mask_bboxes"),
        stitch_train_bboxes=data_state.get("train_mask_bboxes"),
        stitch_log_only_shapes=data_state.get("log_only_stitch_shapes"),
        stitch_log_only_segment_ids=data_state.get("log_only_stitch_segment_ids"),
        stitch_log_only_bboxes=data_state.get("log_only_mask_bboxes"),
        stitch_log_only_downsample=int(getattr(CFG, "stitch_log_only_downsample", getattr(CFG, "stitch_downsample", 1))),
        stitch_log_only_every_n_epochs=int(getattr(CFG, "stitch_log_only_every_n_epochs", 10)),
        stitch_train=bool(getattr(CFG, "stitch_train", False)),
        stitch_train_every_n_epochs=int(getattr(CFG, "stitch_train_every_n_epochs", 1)),
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
        log(f"loading init weights from {run_state['init_ckpt_path']}")
        state_dict = load_state_dict_from_checkpoint(run_state["init_ckpt_path"])
        incompat = model.load_state_dict(state_dict, strict=False)
        missing = len(incompat.missing_keys)
        unexpected = len(incompat.unexpected_keys)
        log(f"loaded init weights (missing_keys={missing}, unexpected_keys={unexpected})")
    if wandb_logger is not None:
        wandb_logger.watch(model, log="all", log_freq=100)
    return model


def build_trainer(args, wandb_logger):
    trainer_logger = wandb_logger if wandb_logger is not None else False
    return pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        log_every_n_steps=10,
        logger=trainer_logger,
        default_root_dir=CFG.outputs_path,
        accumulate_grad_batches=CFG.accumulate_grad_batches,
        precision=args.precision,
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
    if wandb.run is not None:
        wandb.finish()


def validate(trainer, model, data_state, run_state):
    log("starting trainer.validate (stitch_only)")
    trainer.validate(
        model=model,
        dataloaders=data_state["val_loaders"],
        ckpt_path=run_state["resume_ckpt_path"],
    )
    if wandb.run is not None:
        wandb.finish()
