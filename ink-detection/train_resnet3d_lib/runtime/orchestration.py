import argparse
import datetime
import json
import os
import os.path as osp
import uuid

import torch

from train_resnet3d_lib.runtime.checkpointing import resolve_checkpoint_path
from train_resnet3d_lib.config import (
    CFG,
    apply_metadata_hyperparameters,
    cfg_init,
    load_and_validate_base_config,
    log,
    merge_config_with_overrides,
    slugify,
    unflatten_dict,
)
from train_resnet3d_lib.runtime.metadata_config import apply_top_level_stitch_to_cfg
from train_resnet3d_lib.runtime import wandb_runtime
from train_resnet3d_lib.runtime.run_naming import build_default_run_slug
from train_resnet3d_lib.runtime.wandb_local_metrics import LocalMetricsWandbLogger


def _resolve_run_dir(*, cfg, resume_ckpt_path, outputs_path, run_name, run_slug, run_id):
    if resume_ckpt_path and outputs_path is None:
        ckpt_dir = osp.dirname(resume_ckpt_path)
        if osp.basename(ckpt_dir) == "checkpoints":
            inferred_run_dir = osp.dirname(ckpt_dir)
            if osp.isdir(inferred_run_dir):
                return inferred_run_dir, (run_name or osp.basename(inferred_run_dir)), "resume"
    return osp.join(cfg.outputs_path, "runs", f"{slugify(run_slug)}_{run_id}"), run_slug, run_id


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
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--outputs_path", type=str, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Lightning precision mode. Use --precision auto to derive from metadata use_amp.",
    )
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument(
        "--stitch_only",
        action="store_true",
        help="Run validation + stitched logging only (no trainer.fit training loop).",
    )
    return parser.parse_args()


def log_startup(args):
    log(f"start pid={os.getpid()} cwd={os.getcwd()}")
    log(
        "args "
        f"metadata_json={args.metadata_json!r} valid_id={args.valid_id!r} outputs_path={args.outputs_path!r} "
        f"devices={args.devices} accelerator={args.accelerator!r} precision={args.precision!r} "
        f"run_name={args.run_name!r} init_ckpt_path={args.init_ckpt_path!r} "
        f"resume_from_ckpt={args.resume_from_ckpt!r} stitch_only={args.stitch_only}"
    )
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    log(
        f"torch cuda_available={cuda_available} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
        f"device_count={device_count}"
    )

def merge_config(base_config, wandb_logger, args, *, preinit_overrides=None):
    merged_config = merge_config_with_overrides(base_config, preinit_overrides or {})
    wandb_overrides = {}
    if wandb_logger is not None:
        wandb_overrides = unflatten_dict(dict(wandb_logger.experiment.config))
    merged_config = merge_config_with_overrides(merged_config, wandb_overrides)

    apply_metadata_hyperparameters(CFG, merged_config)
    apply_top_level_stitch_to_cfg(CFG, merged_config)
    log("cfg " + json.dumps(merged_config, sort_keys=True, default=str))
    log("args_json " + json.dumps(vars(args), sort_keys=True, default=str))
    if wandb_logger is not None:
        wandb_runtime.sync_wandb_run_config(wandb_logger, merged_config)
    return merged_config


def prepare_runtime_state(
    cfg,
    merged_config,
    *,
    valid_id=None,
    outputs_path=None,
    run_name=None,
    init_ckpt_path=None,
    resume_from_ckpt=None,
):
    segments_metadata = dict(merged_config["segments"])
    if not segments_metadata:
        raise ValueError("metadata_json must define at least one segment under key 'segments'")
    fragment_ids = [str(fragment_id) for fragment_id in segments_metadata.keys()]

    training_cfg = dict(merged_config["training"])
    train_fragment_ids = [str(fragment_id) for fragment_id in list(training_cfg.get("train_segments") or fragment_ids)]
    val_fragment_ids = [str(fragment_id) for fragment_id in list(training_cfg.get("val_segments") or fragment_ids)]
    stitch_target = "all" if bool(getattr(cfg, "stitch_all_val", False)) else str(cfg.valid_id)
    log(
        "segments "
        f"train={len(train_fragment_ids)} val={len(val_fragment_ids)} "
        f"stitch_target={stitch_target!r}"
    )

    missing_train = sorted(set(train_fragment_ids) - set(fragment_ids))
    missing_val = sorted(set(val_fragment_ids) - set(fragment_ids))
    if missing_train:
        raise ValueError(f"training.train_segments contains unknown segment ids: {missing_train}")
    if missing_val:
        raise ValueError(f"training.val_segments contains unknown segment ids: {missing_val}")
    log(f"segments train_ids={train_fragment_ids}")
    log(f"segments val_ids={val_fragment_ids}")

    group_dro_cfg = dict(merged_config["group_dro"])
    group_key = group_dro_cfg["group_key"]

    requested_init_ckpt_path = init_ckpt_path or training_cfg.get("init_ckpt_path") or training_cfg.get("finetune_from")
    if bool(getattr(cfg, "pretrained", True)):
        resolved_init_ckpt_path = resolve_checkpoint_path(requested_init_ckpt_path)
    else:
        if requested_init_ckpt_path:
            log("CFG.pretrained=False; ignoring init_ckpt_path/finetune_from.")
        resolved_init_ckpt_path = None
    cfg.init_ckpt_path = resolved_init_ckpt_path

    resolved_resume_ckpt_path = resolve_checkpoint_path(
        resume_from_ckpt or training_cfg.get("resume_from_ckpt") or training_cfg.get("resume_ckpt_path")
    )
    if resolved_resume_ckpt_path and not osp.exists(resolved_resume_ckpt_path):
        raise FileNotFoundError(f"resume_from_ckpt not found: {resolved_resume_ckpt_path}")
    if resolved_resume_ckpt_path and resolved_init_ckpt_path:
        log("resume_from_ckpt is set; init_ckpt_path will be ignored (resume restores model weights).")

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
    if cfg.objective == "group_dro" and cfg.loss_mode != "per_sample":
        raise ValueError("GroupDRO requires training.loss_mode=per_sample")
    if cfg.objective == "group_dro" and robust_step_size is None:
        raise ValueError("group_dro.robust_step_size is required when training.objective is group_dro")

    if valid_id is not None:
        cfg.valid_id = valid_id
    if cfg.valid_id is not None:
        cfg.valid_id = str(cfg.valid_id)
    if cfg.valid_id is None and val_fragment_ids:
        cfg.valid_id = val_fragment_ids[0]
    if cfg.valid_id not in val_fragment_ids and val_fragment_ids:
        raise ValueError(f"--valid_id {cfg.valid_id!r} is not in training.val_segments")

    if outputs_path is not None:
        cfg.outputs_path = str(outputs_path)

    run_slug = run_name or build_default_run_slug(
        objective=cfg.objective,
        sampler=cfg.sampler,
        loss_mode=cfg.loss_mode,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        cv_fold=getattr(cfg, "cv_fold", None),
    )
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    run_dir, run_slug, run_id = _resolve_run_dir(
        cfg=cfg,
        resume_ckpt_path=resolved_resume_ckpt_path,
        outputs_path=outputs_path,
        run_name=run_name,
        run_slug=run_slug,
        run_id=run_id,
    )
    log(f"run_dir={run_dir}")

    cfg.outputs_path = run_dir
    cfg.model_dir = osp.join(run_dir, "checkpoints")
    cfg.figures_dir = osp.join(run_dir, "figures")
    cfg.submission_dir = osp.join(run_dir, "submissions")
    cfg.log_dir = osp.join(run_dir, "logs")
    cfg_init(cfg)
    log(f"dirs checkpoints={cfg.model_dir} logs={cfg.log_dir}")
    torch.set_float32_matmul_precision("medium")

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
        "init_ckpt_path": resolved_init_ckpt_path,
        "resume_ckpt_path": resolved_resume_ckpt_path,
        "run_slug": run_slug,
        "run_id": run_id,
        "run_dir": run_dir,
    }


def prepare_run(args, merged_config, wandb_logger):
    run_state = prepare_runtime_state(
        CFG,
        merged_config,
        valid_id=args.valid_id,
        outputs_path=args.outputs_path,
        run_name=args.run_name,
        init_ckpt_path=args.init_ckpt_path,
        resume_from_ckpt=args.resume_from_ckpt,
    )
    if wandb_logger is not None:
        if not isinstance(wandb_logger, LocalMetricsWandbLogger):
            raise TypeError(
                "wandb_logger must be LocalMetricsWandbLogger when W&B is enabled, "
                f"got {type(wandb_logger).__name__}"
            )
        wandb_logger.configure_local_persistence(log_dir=str(CFG.log_dir))
        run = wandb_logger.experiment
        run_dir_name = osp.basename(str(run_state["run_dir"]).rstrip("/\\"))
        if not run_dir_name:
            raise ValueError(f"failed to derive run_dir basename from run_dir={run_state['run_dir']!r}")
        desired_run_name = str(run_dir_name)
        current_run_name = str(run.name) if run.name is not None else None
        if current_run_name != desired_run_name:
            run.name = desired_run_name
            log(f"wandb run name updated current={current_run_name!r} merged={desired_run_name!r}")
        run.summary["local/run_dir"] = str(run_state["run_dir"])
        run.summary["local/checkpoints_dir"] = str(CFG.model_dir)
        run.summary["local/log_dir"] = str(CFG.log_dir)
    return run_state
