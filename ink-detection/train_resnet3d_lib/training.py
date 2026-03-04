import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import wandb

from train_resnet3d_lib.config import (
    CFG,
    log,
)
from train_resnet3d_lib.model import RegressionPLModel
from train_resnet3d_lib.modeling.model_config import build_regression_model_configs
from train_resnet3d_lib.runtime.checkpointing import load_state_dict_from_checkpoint
from train_resnet3d_lib.runtime.wandb_local_metrics import LocalMetricsWandbLogger


def _resolve_trainer_precision(cli_precision):
    if cli_precision is not None:
        cli_precision_text = str(cli_precision).strip().lower()
        if cli_precision_text and cli_precision_text != "auto":
            return cli_precision
    use_amp = bool(getattr(CFG, "use_amp", True))
    return "16-mixed" if use_amp else "32-true"


def build_model(run_state, data_state, wandb_logger):
    model_cfg, objective_cfg, stitch_cfg = build_regression_model_configs(run_state, data_state)
    model = RegressionPLModel(
        model_cfg=model_cfg,
        objective_cfg=objective_cfg,
        stitch_cfg=stitch_cfg,
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
