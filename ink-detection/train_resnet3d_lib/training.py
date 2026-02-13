import argparse
import inspect
import json
import os
import os.path as osp
import time
import datetime
import uuid

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb

from train_resnet3d_lib.config import (
    CFG,
    log,
    load_metadata_json,
    deep_merge_dict,
    unflatten_dict,
    apply_metadata_hyperparameters,
    cfg_init,
    slugify,
)
from train_resnet3d_lib.datasets_builder import build_datasets
from train_resnet3d_lib.model import RegressionPLModel
from train_resnet3d_lib.checkpointing import resolve_checkpoint_path, load_state_dict_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_json", type=str, default=None)
    parser.add_argument("--valid_id", type=str, default=None)
    parser.add_argument("--init_ckpt_path", type=str, default=None)
    parser.add_argument(
        "--resume_from_ckpt",
        type=str,
        default=None,
        help="Resume training state (model/optimizer/scheduler/epoch) from a PyTorch Lightning .ckpt.",
    )
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--outputs_path", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    return parser.parse_args()


def log_startup(args):
    log(f"start pid={os.getpid()} cwd={os.getcwd()}")
    log(
        "args "
        f"metadata_json={args.metadata_json!r} valid_id={args.valid_id!r} outputs_path={args.outputs_path!r} "
        f"devices={args.devices} accelerator={args.accelerator!r} precision={args.precision!r} "
        f"run_name={args.run_name!r} init_ckpt_path={args.init_ckpt_path!r} resume_from_ckpt={args.resume_from_ckpt!r}"
    )
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    log(
        f"torch cuda_available={cuda_available} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"device_count={device_count}"
    )


def _warn(message, exc=None):
    if exc is None:
        log(f"WARNING: {message}")
        return
    log(f"WARNING: {message}: {type(exc).__name__}: {exc}")


def load_base_config(args):
    if not args.metadata_json:
        raise ValueError("--metadata_json is required")

    metadata_path = args.metadata_json
    if not osp.isabs(metadata_path):
        if not osp.exists(metadata_path):
            metadata_path = osp.join(osp.dirname(__file__), metadata_path)
    log(f"loading metadata_json={metadata_path}")
    base_config = load_metadata_json(metadata_path)

    if not isinstance(base_config, dict):
        raise TypeError(f"metadata_json root must be an object, got {type(base_config).__name__}")
    if "training" not in base_config or not isinstance(base_config["training"], dict):
        raise KeyError("metadata_json must define an object at key 'training'")
    if "group_dro" not in base_config or not isinstance(base_config["group_dro"], dict):
        raise KeyError("metadata_json must define an object at key 'group_dro'")

    required_training_keys = ("objective", "sampler", "loss_mode", "save_every_epoch")
    for key in required_training_keys:
        if key not in base_config["training"]:
            raise KeyError(f"metadata_json.training missing required key: {key!r}")
    if "group_key" not in base_config["group_dro"]:
        raise KeyError("metadata_json.group_dro missing required key: 'group_key'")
    return base_config


def init_wandb_logger(args, base_config):
    wandb_cfg = base_config.get("wandb", {}) or {}

    wandb_project = args.project
    if wandb_project is None:
        wandb_project = wandb_cfg.get("project") or os.environ.get("WANDB_PROJECT") or "vesuvius"
    wandb_entity = args.entity
    if wandb_entity is None:
        wandb_entity = wandb_cfg.get("entity") or os.environ.get("WANDB_ENTITY")

    wandb_group = args.wandb_group
    if wandb_group is None:
        wandb_group = wandb_cfg.get("group")

    wandb_tags = None
    if args.wandb_tags is not None:
        wandb_tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]
    else:
        tags_cfg = wandb_cfg.get("tags")
        if tags_cfg is not None:
            if isinstance(tags_cfg, str):
                wandb_tags = [t.strip() for t in tags_cfg.split(",") if t.strip()]
            else:
                wandb_tags = list(tags_cfg)

    wandb_logger_kwargs = {}
    if wandb_entity is not None:
        wandb_logger_kwargs["entity"] = wandb_entity
    if wandb_group is not None:
        wandb_logger_kwargs["group"] = wandb_group
    if wandb_tags:
        wandb_logger_kwargs["tags"] = wandb_tags

    try:
        wandb_logger_sig = inspect.signature(WandbLogger.__init__)
    except (TypeError, ValueError) as exc:
        _warn("could not inspect WandbLogger signature; keeping kwargs as-is", exc)
    else:
        wandb_logger_kwargs = {k: v for k, v in wandb_logger_kwargs.items() if k in wandb_logger_sig.parameters}

    log(
        "wandb init "
        f"project={wandb_project!r} entity={wandb_entity!r} group={wandb_group!r} tags={wandb_tags} "
        f"mode={os.environ.get('WANDB_MODE')!r}"
    )
    wandb_t0 = time.time()
    try:
        wandb_logger = WandbLogger(project=wandb_project, name=args.run_name, **wandb_logger_kwargs)
    except Exception as exc:
        _warn("wandb init failed; training will continue with logger disabled", exc)
        return None, {
            "group": wandb_group,
            "tags": wandb_tags,
            "project": wandb_project,
            "entity": wandb_entity,
        }
    log(f"wandb ready in {time.time() - wandb_t0:.1f}s")

    # Sweep selection: ensure W&B summarizes robust metrics by the best value over training,
    # not the last logged value.
    try:
        run = wandb_logger.experiment
        run.define_metric("trainer/global_step")
        run.define_metric("*", step_metric="trainer/global_step", step_sync=True)

        from train_resnet3d_lib.val_stitch_wandb import WANDB_VAL_STITCH_METRIC_SUMMARIES

        metric_summaries = {
            "val/worst_group_loss": "min",
            "val/avg_loss": "min",
            "val/worst_group_dice": "max",
            "val/avg_dice": "max",
            "val/worst_group_loss_ema": "min",
            "val/avg_loss_ema": "min",
            "val/worst_group_dice_ema": "max",
            "val/avg_dice_ema": "max",
            "train/total_loss_ema": "min",
            "train/dice_ema": "max",
            "train/worst_group_loss_ema": "min",
            "metrics/val/dice": "max",
        }
        overlap = set(metric_summaries) & set(WANDB_VAL_STITCH_METRIC_SUMMARIES)
        if overlap:
            raise ValueError(f"wandb metric summary keys overlap: {sorted(overlap)!r}")
        metric_summaries.update(WANDB_VAL_STITCH_METRIC_SUMMARIES)
        for metric_name, summary_mode in metric_summaries.items():
            run.define_metric(metric_name, summary=summary_mode)
    except Exception as exc:
        _warn("wandb metric definitions failed; continuing", exc)

    wandb_info = {
        "group": wandb_group,
        "tags": wandb_tags,
        "project": wandb_project,
        "entity": wandb_entity,
    }
    return wandb_logger, wandb_info


def merge_config(base_config, wandb_logger, args):
    wandb_overrides = {}
    if wandb_logger is not None:
        try:
            wandb_overrides = unflatten_dict(dict(wandb_logger.experiment.config))
        except Exception as exc:
            _warn("failed to parse wandb config overrides; proceeding with metadata/base config only", exc)

    merged_config = json.loads(json.dumps(base_config))
    deep_merge_dict(merged_config, wandb_overrides)

    apply_metadata_hyperparameters(CFG, merged_config)
    log("cfg " + json.dumps(merged_config, sort_keys=True, default=str))
    log("args_json " + json.dumps(vars(args), sort_keys=True, default=str))

    if wandb_logger is not None:
        try:
            wandb_logger.experiment.config.update(merged_config, allow_val_change=True)
        except Exception as exc:
            _warn("failed to sync merged config back to wandb", exc)

    return merged_config


def prepare_run(args, merged_config, wandb_logger, wandb_info):
    if "segments" not in merged_config or not isinstance(merged_config["segments"], dict):
        raise KeyError("metadata_json must define an object at key 'segments'")
    segments_metadata = merged_config["segments"]
    if not segments_metadata:
        raise ValueError("metadata_json must define at least one segment under key 'segments'")
    fragment_ids = list(segments_metadata.keys())

    if "training" not in merged_config or not isinstance(merged_config["training"], dict):
        raise KeyError("metadata_json must define an object at key 'training'")
    training_cfg = merged_config["training"]
    train_fragment_ids = training_cfg.get("train_segments")
    val_fragment_ids = training_cfg.get("val_segments")
    if train_fragment_ids is None:
        train_fragment_ids = fragment_ids
    if val_fragment_ids is None:
        val_fragment_ids = fragment_ids
    train_fragment_ids = list(train_fragment_ids)
    val_fragment_ids = list(val_fragment_ids)
    stitch_target = "all" if bool(getattr(CFG, "stitch_all_val", False)) else str(CFG.valid_id)
    log(
        "segments "
        f"train={len(train_fragment_ids)} val={len(val_fragment_ids)} "
        f"stitch_target={stitch_target!r}"
    )

    if segments_metadata:
        missing_train = sorted(set(train_fragment_ids) - set(fragment_ids))
        missing_val = sorted(set(val_fragment_ids) - set(fragment_ids))
        if missing_train:
            raise ValueError(f"training.train_segments contains unknown segment ids: {missing_train}")
        if missing_val:
            raise ValueError(f"training.val_segments contains unknown segment ids: {missing_val}")

    if "group_dro" not in merged_config or not isinstance(merged_config["group_dro"], dict):
        raise KeyError("metadata_json must define an object at key 'group_dro'")
    group_dro_cfg = merged_config["group_dro"]
    group_key = group_dro_cfg.get("group_key", "base_path")

    init_ckpt_path = resolve_checkpoint_path(
        args.init_ckpt_path or training_cfg.get("init_ckpt_path") or training_cfg.get("finetune_from")
    )
    CFG.init_ckpt_path = init_ckpt_path

    resume_ckpt_path = resolve_checkpoint_path(
        args.resume_from_ckpt or training_cfg.get("resume_from_ckpt") or training_cfg.get("resume_ckpt_path")
    )
    if resume_ckpt_path:
        if not osp.exists(resume_ckpt_path):
            raise FileNotFoundError(f"resume_from_ckpt not found: {resume_ckpt_path}")

    if resume_ckpt_path and init_ckpt_path:
        log("resume_from_ckpt is set; init_ckpt_path will be ignored (resume restores model weights).")

    if CFG.objective not in {"erm", "group_dro"}:
        raise ValueError(f"Unknown training.objective: {CFG.objective!r}")
    if CFG.sampler not in {"shuffle", "group_balanced", "group_stratified"}:
        raise ValueError(f"Unknown training.sampler: {CFG.sampler!r}")
    if CFG.loss_mode not in {"batch", "per_sample"}:
        raise ValueError(f"Unknown training.loss_mode: {CFG.loss_mode!r}")

    robust_step_size = group_dro_cfg.get("robust_step_size")
    group_dro_gamma = group_dro_cfg.get("gamma", 0.1)
    group_dro_btl = bool(group_dro_cfg.get("btl", False))
    group_dro_alpha = group_dro_cfg.get("alpha")
    group_dro_normalize_loss = bool(group_dro_cfg.get("normalize_loss", False))
    group_dro_min_var_weight = group_dro_cfg.get(
        "minimum_variational_weight",
        group_dro_cfg.get("min_var_weight", 0.0),
    )
    group_dro_adj = group_dro_cfg.get("adj")
    log(
        "group_dro "
        f"group_key={group_key!r} robust_step_size={robust_step_size!r} "
        f"gamma={group_dro_gamma} btl={group_dro_btl} alpha={group_dro_alpha!r} normalize_loss={group_dro_normalize_loss}"
    )

    if CFG.objective == "group_dro" and CFG.loss_mode != "per_sample":
        raise ValueError("GroupDRO requires training.loss_mode=per_sample")
    if CFG.objective == "group_dro" and robust_step_size is None:
        raise ValueError("group_dro.robust_step_size is required when training.objective is group_dro")

    if args.valid_id is not None:
        CFG.valid_id = args.valid_id
    if args.valid_id is not None and CFG.valid_id not in val_fragment_ids and val_fragment_ids:
        raise ValueError(f"--valid_id {CFG.valid_id!r} is not in training.val_segments")
    if CFG.valid_id not in val_fragment_ids and val_fragment_ids:
        CFG.valid_id = val_fragment_ids[0]

    if args.outputs_path is not None:
        CFG.outputs_path = str(args.outputs_path)

    if args.run_name is None:
        lr_tag = f"{float(CFG.lr):.2e}"
        wd_tag = f"{float(CFG.weight_decay):.2e}"
        run_slug = (
            f"{CFG.objective}_{CFG.sampler}_{CFG.loss_mode}_"
            f"lr={lr_tag}_wd={wd_tag}"
        )
    else:
        run_slug = args.run_name
    if args.run_name is None and getattr(CFG, "cv_fold", None) is not None:
        run_slug = f"{run_slug}_fold={CFG.cv_fold}"
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir = None

    if resume_ckpt_path and args.outputs_path is None:
        # Resume in-place if resuming from one of our checkpoints:
        #   .../runs/<run>/checkpoints/<file>.ckpt
        ckpt_dir = osp.dirname(resume_ckpt_path)
        if osp.basename(ckpt_dir) == "checkpoints":
            inferred_run_dir = osp.dirname(ckpt_dir)
            if osp.isdir(inferred_run_dir):
                run_dir = inferred_run_dir
                run_slug = args.run_name or osp.basename(run_dir)
                run_id = "resume"

    if run_dir is None:
        run_dir = osp.join(CFG.outputs_path, "runs", f"{slugify(run_slug)}_{run_id}")
    log(f"run_dir={run_dir}")

    cv_fold = getattr(CFG, "cv_fold", None)
    if cv_fold is not None:
        fold_tag = f"fold{cv_fold}"
        tags = list(wandb_info.get("tags") or [])
        if fold_tag not in tags:
            tags.append(fold_tag)
        wandb_info["tags"] = tags

    if wandb_logger is not None:
        try:
            if wandb_info.get("group") is not None:
                wandb_logger.experiment.group = str(wandb_info["group"])
            if wandb_info.get("tags"):
                wandb_logger.experiment.tags = wandb_info["tags"]
            if args.run_name is None:
                wandb_logger.experiment.name = f"{run_slug}_{run_id}"
        except Exception as exc:
            _warn("failed to apply wandb run metadata (group/tags/name)", exc)

    CFG.outputs_path = run_dir
    CFG.model_dir = osp.join(run_dir, "checkpoints")
    CFG.figures_dir = osp.join(run_dir, "figures")
    CFG.submission_dir = osp.join(run_dir, "submissions")
    CFG.log_dir = osp.join(run_dir, "logs")

    cfg_init(CFG)
    log(f"dirs checkpoints={CFG.model_dir} logs={CFG.log_dir}")

    torch.set_float32_matmul_precision('medium')

    return {
        "segments_metadata": segments_metadata,
        "fragment_ids": fragment_ids,
        "train_fragment_ids": train_fragment_ids,
        "val_fragment_ids": val_fragment_ids,
        "group_dro_cfg": group_dro_cfg,
        "group_key": group_key,
        "robust_step_size": robust_step_size,
        "group_dro_gamma": group_dro_gamma,
        "group_dro_btl": group_dro_btl,
        "group_dro_alpha": group_dro_alpha,
        "group_dro_normalize_loss": group_dro_normalize_loss,
        "group_dro_min_var_weight": group_dro_min_var_weight,
        "group_dro_adj": group_dro_adj,
        "init_ckpt_path": init_ckpt_path,
        "resume_ckpt_path": resume_ckpt_path,
    }


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
        try:
            wandb_logger.watch(model, log="all", log_freq=100)
        except Exception as exc:
            _warn("wandb watch failed; continuing without parameter/gradient watching", exc)
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
                        every_n_epochs=1,
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


def run(args):
    log_startup(args)
    base_config = load_base_config(args)
    wandb_logger, wandb_info = init_wandb_logger(args, base_config)
    merged_config = merge_config(base_config, wandb_logger, args)
    run_state = prepare_run(args, merged_config, wandb_logger, wandb_info)
    data_state = build_datasets(run_state)
    model = build_model(run_state, data_state, wandb_logger)
    trainer = build_trainer(args, wandb_logger)
    fit(trainer, model, data_state, run_state)


def main():
    run(parse_args())
