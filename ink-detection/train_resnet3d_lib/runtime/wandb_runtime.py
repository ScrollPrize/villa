import inspect
import os
import os.path as osp
import time

import wandb
import yaml

from train_resnet3d_lib.config import (
    CFG,
    log,
    merge_config_with_overrides,
    unflatten_dict,
)
from train_resnet3d_lib.runtime.wandb_local_metrics import LocalMetricsWandbLogger
from train_resnet3d_lib.runtime.run_naming import build_default_run_slug_from_metadata


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
        if "segments/*/" in metric_name:
            if not safe_segment_ids:
                raise ValueError("cannot expand wildcard metric summary keys without any segment ids")
            for safe_segment_id in safe_segment_ids:
                expanded_key = metric_name.replace("*", safe_segment_id)
                expanded[expanded_key] = summary_mode
            continue
        if metric_name.endswith("/thr_*"):
            expanded[metric_name] = summary_mode
            continue
        raise ValueError(f"unsupported wildcard metric summary key: {metric_name!r}")
    return expanded


def init_wandb_logger(args, base_config, *, preinit_overrides=None):
    init_config = merge_config_with_overrides(base_config, preinit_overrides or {})
    wandb_cfg = dict(init_config["wandb"])
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

    wandb_logger_sig = inspect.signature(LocalMetricsWandbLogger.__init__)
    wandb_logger_kwargs = {k: v for k, v in wandb_logger_kwargs.items() if k in wandb_logger_sig.parameters}

    initial_run_name = args.run_name
    if initial_run_name is None:
        initial_run_name = build_default_run_slug_from_metadata(init_config)
    log(
        "wandb init "
        f"project={wandb_project!r} entity={wandb_entity!r} group={wandb_group!r} tags={wandb_tags} "
        f"name={initial_run_name!r} mode={os.environ.get('WANDB_MODE')!r}"
    )
    wandb_t0 = time.time()
    wandb_logger = LocalMetricsWandbLogger(project=wandb_project, name=initial_run_name, **wandb_logger_kwargs)
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
        "metrics/val/dice_hist_thr_96_255": "max",
        "metrics/val/balanced_accuracy_hist_thr_96_255": "max",
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
        if "*" in metric_name and not metric_name.endswith("/thr_*"):
            raise ValueError(f"wildcard metric key reached define_metric: {metric_name!r}")
        run.define_metric(
            metric_name,
            summary=summary_mode,
            step_metric="trainer/global_step",
            step_sync=True,
        )


def sync_wandb_run_config(wandb_logger, merged_config):
    merged_wandb_cfg = dict(merged_config["wandb"])
    merged_tags = tuple(merged_wandb_cfg["tags"])
    run = wandb_logger.experiment
    current_tags = tuple(run.tags) if run.tags is not None else ()
    if current_tags != merged_tags:
        run.tags = merged_tags
        log(f"wandb tags updated current={list(current_tags)!r} merged={list(merged_tags)!r}")
    wandb_logger.experiment.config.update(merged_config, allow_val_change=True)
    define_wandb_metric_summaries(wandb_logger, merged_config)


def configure_wandb_run(wandb_logger, *, run_state, cfg):
    if wandb_logger is None:
        return
    if not isinstance(wandb_logger, LocalMetricsWandbLogger):
        raise TypeError(
            "wandb_logger must be LocalMetricsWandbLogger when W&B is enabled, "
            f"got {type(wandb_logger).__name__}"
        )

    wandb_logger.configure_local_persistence(log_dir=str(cfg.log_dir))
    run = wandb_logger.experiment
    run_dir = str(run_state["run_dir"])
    run_dir_name = osp.basename(run_dir.rstrip("/\\"))
    if not run_dir_name:
        raise ValueError(f"failed to derive run_dir basename from run_dir={run_dir!r}")
    desired_run_name = str(run_dir_name)
    current_run_name = str(run.name) if run.name is not None else None
    if current_run_name != desired_run_name:
        run.name = desired_run_name
        log(f"wandb run name updated current={current_run_name!r} merged={desired_run_name!r}")
    run.summary["local/run_dir"] = run_dir
    run.summary["local/checkpoints_dir"] = str(cfg.model_dir)
    run.summary["local/log_dir"] = str(cfg.log_dir)


def finalize_wandb_logging(trainer):
    logger = getattr(trainer, "logger", None)
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
