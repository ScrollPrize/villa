import argparse
import datetime
import inspect
import json
import os
import os.path as osp
import time
import uuid
import yaml

import torch
from pytorch_lightning.loggers import WandbLogger

from train_resnet3d_lib.checkpointing import resolve_checkpoint_path
from train_resnet3d_lib.config import (
    CFG,
    apply_metadata_hyperparameters,
    cfg_init,
    load_and_validate_base_config,
    log,
    merge_config_with_overrides,
    normalize_cv_fold,
    normalize_wandb_config,
    slugify,
    unflatten_dict,
)


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
    parser.add_argument("--precision", type=str, default="16-mixed")
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


def load_base_config(args):
    return load_and_validate_base_config(args.metadata_json, base_dir=os.path.dirname(__file__))


def _normalize_wandb_sweep_param_value(raw_value):
    if isinstance(raw_value, dict) and "value" in raw_value:
        return raw_value["value"]
    if isinstance(raw_value, dict) and "values" in raw_value and len(raw_value) == 1:
        values = raw_value["values"]
        if isinstance(values, list) and len(values) == 1:
            return values[0]
    return raw_value


def load_wandb_preinit_overrides():
    sweep_param_path = os.environ.get("WANDB_SWEEP_PARAM_PATH")
    if sweep_param_path is None:
        return {}
    if not osp.exists(sweep_param_path):
        raise FileNotFoundError(
            "WANDB_SWEEP_PARAM_PATH is set but file was not found: "
            f"{sweep_param_path!r}"
        )
    with open(sweep_param_path, "r") as f:
        sweep_params = yaml.safe_load(f)
    if sweep_params is None:
        return {}
    if not isinstance(sweep_params, dict):
        raise TypeError(
            "sweep parameter file must contain an object mapping parameter names to values, "
            f"got {type(sweep_params).__name__}"
        )
    flat_overrides = {}
    for param_key, raw_value in sweep_params.items():
        if not isinstance(param_key, str):
            raise TypeError(
                "sweep parameter keys must be strings, "
                f"got {type(param_key).__name__}: {param_key!r}"
            )
        flat_overrides[param_key] = _normalize_wandb_sweep_param_value(raw_value)
    if flat_overrides:
        log(
            "wandb preinit overrides "
            f"path={sweep_param_path!r} keys={sorted(flat_overrides.keys())!r}"
        )
    return unflatten_dict(flat_overrides)


def build_default_run_slug(*, objective, sampler, loss_mode, lr, weight_decay, cv_fold=None):
    lr_tag = f"{float(lr):.2e}"
    wd_tag = f"{float(weight_decay):.2e}"
    run_slug = f"{objective}_{sampler}_{loss_mode}_lr={lr_tag}_wd={wd_tag}"
    normalized_cv_fold = normalize_cv_fold(cv_fold)
    if normalized_cv_fold is not None:
        run_slug = f"{run_slug}_fold={normalized_cv_fold}"
    return run_slug


def infer_default_run_slug(merged_config):
    if "training" not in merged_config or not isinstance(merged_config["training"], dict):
        raise KeyError("merged_config must define an object at key 'training'")
    if "training_hyperparameters" not in merged_config or not isinstance(merged_config["training_hyperparameters"], dict):
        raise KeyError("merged_config must define an object at key 'training_hyperparameters'")
    hp_cfg = merged_config["training_hyperparameters"]
    if "training" not in hp_cfg or not isinstance(hp_cfg["training"], dict):
        raise KeyError("merged_config.training_hyperparameters must define an object at key 'training'")

    training_cfg = merged_config["training"]
    hp_train_cfg = hp_cfg["training"]
    return build_default_run_slug(
        objective=str(training_cfg["objective"]).lower(),
        sampler=str(training_cfg["sampler"]).lower(),
        loss_mode=str(training_cfg["loss_mode"]).lower(),
        lr=float(hp_train_cfg["lr"]),
        weight_decay=float(hp_train_cfg["weight_decay"]),
        cv_fold=training_cfg.get("cv_fold"),
    )


def expand_wandb_metric_summary_keys(metric_summaries, *, segment_ids):
    safe_segment_ids = []
    safe_to_original = {}
    for segment_id in segment_ids:
        original = str(segment_id)
        safe_segment_id = original.replace("/", "_")
        if safe_segment_id in safe_to_original and safe_to_original[safe_segment_id] != original:
            raise ValueError(
                "segment id collision after sanitization for W&B metric keys: "
                f"{safe_to_original[safe_segment_id]!r} and {original!r} -> {safe_segment_id!r}"
            )
        safe_to_original[safe_segment_id] = original
        safe_segment_ids.append(safe_segment_id)

    expanded = {}
    for metric_name, summary_mode in metric_summaries.items():
        if "*" not in metric_name:
            expanded[metric_name] = summary_mode
            continue
        if metric_name.count("*") != 1:
            raise ValueError(f"unsupported wildcard metric summary key: {metric_name!r}")
        if not safe_segment_ids:
            raise ValueError("cannot expand wildcard metric summary keys without any segment ids")
        for safe_segment_id in safe_segment_ids:
            expanded_key = metric_name.replace("*", safe_segment_id)
            expanded[expanded_key] = summary_mode
    return expanded


def init_wandb_logger(args, base_config, *, preinit_overrides=None):
    init_config = merge_config_with_overrides(base_config, preinit_overrides or {})
    wandb_cfg = normalize_wandb_config(init_config["wandb"], key_prefix="metadata_json.wandb")
    wandb_enabled = wandb_cfg["enabled"]
    wandb_project = wandb_cfg["project"]
    wandb_entity = wandb_cfg["entity"]
    wandb_group = wandb_cfg["group"]
    wandb_tags = list(wandb_cfg["tags"])

    wandb_logger_kwargs = {"entity": wandb_entity}
    if wandb_group is not None:
        wandb_logger_kwargs["group"] = wandb_group
    if wandb_tags:
        wandb_logger_kwargs["tags"] = wandb_tags

    if not wandb_enabled:
        log("wandb disabled")
        return None

    wandb_logger_sig = inspect.signature(WandbLogger.__init__)
    wandb_logger_kwargs = {k: v for k, v in wandb_logger_kwargs.items() if k in wandb_logger_sig.parameters}

    initial_run_name = args.run_name
    if initial_run_name is None:
        initial_run_name = infer_default_run_slug(init_config)
    log(
        "wandb init "
        f"project={wandb_project!r} entity={wandb_entity!r} group={wandb_group!r} tags={wandb_tags} "
        f"name={initial_run_name!r} mode={os.environ.get('WANDB_MODE')!r}"
    )
    wandb_t0 = time.time()
    wandb_logger = WandbLogger(project=wandb_project, name=initial_run_name, **wandb_logger_kwargs)
    log(f"wandb ready in {time.time() - wandb_t0:.1f}s")
    return wandb_logger


def define_wandb_metric_summaries(wandb_logger, merged_config):
    run = wandb_logger.experiment
    run.define_metric("trainer/global_step")

    from train_resnet3d_lib.val_stitch_wandb import get_wandb_val_stitch_metric_summaries

    val_stitch_metric_summaries = get_wandb_val_stitch_metric_summaries(
        enable_skeleton_metrics=bool(getattr(CFG, "eval_enable_skeleton_metrics", True))
    )

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
    overlap = set(metric_summaries) & set(val_stitch_metric_summaries)
    if overlap:
        raise ValueError(f"wandb metric summary keys overlap: {sorted(overlap)!r}")
    metric_summaries.update(val_stitch_metric_summaries)
    if "segments" not in merged_config or not isinstance(merged_config["segments"], dict):
        raise KeyError("metadata_json must define an object at key 'segments'")
    metric_summaries = expand_wandb_metric_summary_keys(
        metric_summaries,
        segment_ids=list(merged_config["segments"].keys()),
    )
    for metric_name, summary_mode in metric_summaries.items():
        if "*" in metric_name:
            raise ValueError(f"wildcard metric key reached define_metric: {metric_name!r}")
        run.define_metric(
            metric_name,
            summary=summary_mode,
            step_metric="trainer/global_step",
            step_sync=True,
        )


def merge_config(base_config, wandb_logger, args, *, preinit_overrides=None):
    merged_config = merge_config_with_overrides(base_config, preinit_overrides or {})
    wandb_overrides = {}
    if wandb_logger is not None:
        wandb_overrides = unflatten_dict(dict(wandb_logger.experiment.config))
    merged_config = merge_config_with_overrides(merged_config, wandb_overrides)

    apply_metadata_hyperparameters(CFG, merged_config)
    log("cfg " + json.dumps(merged_config, sort_keys=True, default=str))
    log("args_json " + json.dumps(vars(args), sort_keys=True, default=str))
    if wandb_logger is not None:
        merged_wandb_cfg = normalize_wandb_config(merged_config["wandb"], key_prefix="merged_config.wandb")
        merged_tags = tuple(merged_wandb_cfg["tags"])
        run = wandb_logger.experiment
        current_tags = tuple(run.tags) if run.tags is not None else ()
        if current_tags != merged_tags:
            run.tags = merged_tags
            log(f"wandb tags updated current={list(current_tags)!r} merged={list(merged_tags)!r}")
        wandb_logger.experiment.config.update(merged_config, allow_val_change=True)
        define_wandb_metric_summaries(wandb_logger, merged_config)
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
    if "segments" not in merged_config or not isinstance(merged_config["segments"], dict):
        raise KeyError("metadata_json must define an object at key 'segments'")
    segments_metadata = merged_config["segments"]
    if not segments_metadata:
        raise ValueError("metadata_json must define at least one segment under key 'segments'")
    fragment_ids = list(segments_metadata.keys())

    if "training" not in merged_config or not isinstance(merged_config["training"], dict):
        raise KeyError("metadata_json must define an object at key 'training'")
    training_cfg = merged_config["training"]
    train_fragment_ids = list(training_cfg.get("train_segments") or fragment_ids)
    val_fragment_ids = list(training_cfg.get("val_segments") or fragment_ids)
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

    if "group_dro" not in merged_config or not isinstance(merged_config["group_dro"], dict):
        raise KeyError("metadata_json must define an object at key 'group_dro'")
    group_dro_cfg = merged_config["group_dro"]
    group_key = group_dro_cfg["group_key"]

    if "wandb" not in merged_config or not isinstance(merged_config["wandb"], dict):
        raise KeyError("metadata_json must define an object at key 'wandb'")

    resolved_init_ckpt_path = resolve_checkpoint_path(
        init_ckpt_path or training_cfg.get("init_ckpt_path") or training_cfg.get("finetune_from")
    )
    cfg.init_ckpt_path = resolved_init_ckpt_path

    resolved_resume_ckpt_path = resolve_checkpoint_path(
        resume_from_ckpt or training_cfg.get("resume_from_ckpt") or training_cfg.get("resume_ckpt_path")
    )
    if resolved_resume_ckpt_path and not osp.exists(resolved_resume_ckpt_path):
        raise FileNotFoundError(f"resume_from_ckpt not found: {resolved_resume_ckpt_path}")
    if resolved_resume_ckpt_path and resolved_init_ckpt_path:
        log("resume_from_ckpt is set; init_ckpt_path will be ignored (resume restores model weights).")

    if cfg.objective not in {"erm", "group_dro"}:
        raise ValueError(f"Unknown training.objective: {cfg.objective!r}")
    if cfg.sampler not in {"shuffle", "group_balanced", "group_stratified"}:
        raise ValueError(f"Unknown training.sampler: {cfg.sampler!r}")
    if cfg.loss_mode not in {"batch", "per_sample"}:
        raise ValueError(f"Unknown training.loss_mode: {cfg.loss_mode!r}")

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
    if valid_id is not None and cfg.valid_id not in val_fragment_ids and val_fragment_ids:
        raise ValueError(f"--valid_id {cfg.valid_id!r} is not in training.val_segments")
    if cfg.valid_id not in val_fragment_ids and val_fragment_ids:
        cfg.valid_id = val_fragment_ids[0]

    if outputs_path is not None:
        cfg.outputs_path = str(outputs_path)

    if run_name is None:
        run_slug = build_default_run_slug(
            objective=cfg.objective,
            sampler=cfg.sampler,
            loss_mode=cfg.loss_mode,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            cv_fold=getattr(cfg, "cv_fold", None),
        )
    else:
        run_slug = run_name
    run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    run_dir = None
    if resolved_resume_ckpt_path and outputs_path is None:
        ckpt_dir = osp.dirname(resolved_resume_ckpt_path)
        if osp.basename(ckpt_dir) == "checkpoints":
            inferred_run_dir = osp.dirname(ckpt_dir)
            if osp.isdir(inferred_run_dir):
                run_dir = inferred_run_dir
                run_slug = run_name or osp.basename(run_dir)
                run_id = "resume"
    if run_dir is None:
        run_dir = osp.join(cfg.outputs_path, "runs", f"{slugify(run_slug)}_{run_id}")
    log(f"run_dir={run_dir}")

    cfg.outputs_path = run_dir
    cfg.model_dir = osp.join(run_dir, "checkpoints")
    cfg.figures_dir = osp.join(run_dir, "figures")
    cfg.submission_dir = osp.join(run_dir, "submissions")
    cfg.log_dir = osp.join(run_dir, "logs")
    cfg_init(cfg)
    log(f"dirs checkpoints={cfg.model_dir} logs={cfg.log_dir}")
    torch.set_float32_matmul_precision("medium")

    run_state = {
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
    }
    return {"run_state": run_state}


def prepare_run(args, merged_config, wandb_logger):
    prepared = prepare_runtime_state(
        CFG,
        merged_config,
        valid_id=args.valid_id,
        outputs_path=args.outputs_path,
        run_name=args.run_name,
        init_ckpt_path=args.init_ckpt_path,
        resume_from_ckpt=args.resume_from_ckpt,
    )
    run_state = prepared["run_state"]
    if wandb_logger is not None:
        run = wandb_logger.experiment
        desired_run_name = str(run_state["run_slug"])
        current_run_name = str(run.name) if run.name is not None else None
        if current_run_name != desired_run_name:
            run.name = desired_run_name
            log(f"wandb run name updated current={current_run_name!r} merged={desired_run_name!r}")
    return run_state
