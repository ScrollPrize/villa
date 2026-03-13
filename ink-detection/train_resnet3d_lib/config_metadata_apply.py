from train_resnet3d_lib.data.augmentations import rebuild_augmentations


MODEL_SCALARS = {
    "model_name": str,
    "backbone": str,
    "backbone_pretrained_path": str,
    "resnet3d_model_depth": int,
    "in_chans": int,
    "encoder_depth": int,
    "norm": str,
    "group_norm_groups": int,
    "target_size": int,
}

TRAINING_SCALARS = {
    "size": int,
    "tile_size": int,
    "stride": int,
    "train_batch_size": int,
    "valid_batch_size": int,
    "accumulate_grad_batches": int,
    "epochs": int,
    "scheduler": str,
    "optimizer": str,
    "adamw_beta2": float,
    "adamw_eps": float,
    "sgd_momentum": float,
    "warmup_factor": float,
    "lr": float,
    "onecycle_pct_start": float,
    "onecycle_div_factor": float,
    "onecycle_final_div_factor": float,
    "cosine_warmup_pct": float,
    "min_lr": float,
    "weight_decay": float,
    "max_clip_value": object,
    "normalization_mode": str,
    "max_grad_norm": float,
    "patch_loss_weight": float,
    "stitch_loss_weight": float,
    "stitch_boundary_loss_weight": float,
    "stitch_cldice_loss_weight": float,
    "stitch_cldice_mask_mode": str,
    "stitch_betti_matching_loss_weight": float,
    "stitch_betti_matching_filtration_type": str,
    "stitch_betti_matching_num_processes": int,
    "stitch_patch_batch_size": int,
    "bce_smooth_factor": float,
    "soft_label_positive": float,
    "soft_label_negative": float,
    "eval_threshold": float,
    "eval_topological_metrics_every_n_epochs": int,
    "eval_drd_block_size": int,
    "eval_boundary_k": int,
    "eval_boundary_tols": list,
    "eval_skeleton_thinning_type": str,
    "eval_component_worst_q": float,
    "eval_component_worst_k": int,
    "eval_component_min_area": int,
    "eval_component_pad": int,
    "eval_save_stitch_debug_images_every_n_epochs": int,
    "eval_threshold_grid_min": float,
    "eval_threshold_grid_max": float,
    "eval_threshold_grid_steps": int,
    "eval_threshold_grid": object,
    "eval_wandb_media_downsample": int,
    "num_workers": int,
    "layer_read_workers": int,
    "seed": int,
    "dataset_root": str,
    "erm_group_topk": int,
    "save_every_n_epochs": int,
    "data_backend": str,
    "dataset_cache_dir": str,
}

TRAINING_BOOLS = {
    "use_amp",
    "sgd_nesterov",
    "exclude_weight_decay_bias_norm",
    "eval_stitch_metrics",
    "eval_enable_skeleton_metrics",
    "eval_stitch_full_region_metrics",
    "eval_save_stitch_debug_images",
    "save_every_epoch",
    "pretrained",
    "dataset_cache_enabled",
    "dataset_cache_check_hash",
    "stitch_gradient_checkpointing",
    "stitch_save_on_cpu",
}

TRAINING_LOWER = {
    "objective",
    "sampler",
    "group_stratified_epoch_size_mode",
    "loss_mode",
    "loss_recipe",
    "stitch_cldice_mask_mode",
    "stitch_betti_matching_filtration_type",
}

def _section(parent, key, *, key_prefix):
    value = parent.get(key)
    if value is not None and not isinstance(value, dict):
        raise TypeError(f"{key_prefix}.{key} must be an object, got {type(value).__name__}")
    return {} if value is None else value


def _apply_scalar_fields(cfg, source, *, casts):
    for key, caster in casts.items():
        if key not in source:
            continue
        value = source[key]
        if caster is int:
            setattr(cfg, key, int(value))
        elif caster is float:
            setattr(cfg, key, float(value))
        elif caster is str:
            setattr(cfg, key, str(value))
        elif caster is list:
            if not isinstance(value, (list, tuple)):
                raise ValueError(f"{key} must be a list")
            setattr(cfg, key, list(value))
        else:
            setattr(cfg, key, value)


def _apply_bool_fields(cfg, source, *, keys, parse_bool, key_prefix):
    for key in keys:
        if key not in source:
            continue
        setattr(cfg, key, parse_bool(source[key], key=f"{key_prefix}.{key}"))


def _normalize_string_list(value, *, key):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{key} must be a list")
    normalized = []
    for item in value:
        item_text = str(item).strip()
        if not item_text:
            continue
        normalized.append(item_text)
    return normalized
def _apply_stitch_fields(cfg, stitch_cfg, *, parse_bool):
    if "all_val" in stitch_cfg:
        cfg.stitch_all_val = parse_bool(stitch_cfg["all_val"], key="stitch.all_val")
    if "train" in stitch_cfg:
        cfg.stitch_train = parse_bool(stitch_cfg["train"], key="stitch.train")
    if "use_roi" in stitch_cfg:
        cfg.stitch_use_roi = parse_bool(stitch_cfg["use_roi"], key="stitch.use_roi")
    if "downsample" in stitch_cfg:
        cfg.stitch_downsample = int(stitch_cfg["downsample"])
    if "train_every_n_epochs" in stitch_cfg:
        cfg.stitch_train_every_n_epochs = int(stitch_cfg["train_every_n_epochs"])
    if "eval_every_n_epochs" in stitch_cfg:
        cfg.eval_stitch_every_n_epochs = int(stitch_cfg["eval_every_n_epochs"])
    if "eval_plus_one" in stitch_cfg:
        cfg.eval_stitch_every_n_epochs_plus_one = parse_bool(
            stitch_cfg["eval_plus_one"],
            key="stitch.eval_plus_one",
        )
    if "log_only_segments" in stitch_cfg:
        cfg.stitch_log_only_segments = _normalize_string_list(
            stitch_cfg["log_only_segments"],
            key="stitch.log_only_segments",
        )
    if "log_only_every_n_epochs" in stitch_cfg:
        cfg.stitch_log_only_every_n_epochs = int(stitch_cfg["log_only_every_n_epochs"])
    if "log_only_downsample" in stitch_cfg:
        cfg.stitch_log_only_downsample = int(stitch_cfg["log_only_downsample"])


def apply_metadata_hyperparameters(
    cfg,
    metadata,
    *,
    parse_bool,
    parse_optional_positive_int,
    parse_normalization_mode,
    normalize_cv_fold,
):
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be an object, got {type(metadata).__name__}")

    model_cfg = _section(metadata, "model", key_prefix="metadata")
    training_cfg = _section(metadata, "training", key_prefix="metadata")
    augmentation_cfg = _section(metadata, "augmentation", key_prefix="metadata")
    stitch_cfg = _section(metadata, "stitch", key_prefix="metadata")

    _apply_scalar_fields(cfg, model_cfg, casts=MODEL_SCALARS)
    _apply_scalar_fields(cfg, training_cfg, casts=TRAINING_SCALARS)
    _apply_bool_fields(
        cfg,
        training_cfg,
        keys=TRAINING_BOOLS,
        parse_bool=parse_bool,
        key_prefix="training",
    )
    for key in TRAINING_LOWER:
        if key not in training_cfg:
            continue
        setattr(cfg, key, str(training_cfg[key]).strip().lower())

    _apply_stitch_fields(cfg, stitch_cfg, parse_bool=parse_bool)

    if "train_batch_size" in training_cfg and "valid_batch_size" not in training_cfg:
        cfg.valid_batch_size = int(cfg.train_batch_size)
    if "stitch_patch_batch_size" not in training_cfg:
        cfg.stitch_patch_batch_size = int(cfg.valid_batch_size)

    cfg.norm = str(cfg.norm).strip().lower()
    cfg.optimizer = str(cfg.optimizer).strip().lower()
    cfg.data_backend = str(cfg.data_backend).strip().lower()
    cfg.loss_recipe = str(getattr(cfg, "loss_recipe", "dice_bce")).strip().lower()
    cfg.bce_smooth_factor = float(getattr(cfg, "bce_smooth_factor", 0.25))
    cfg.soft_label_positive = float(getattr(cfg, "soft_label_positive", 1.0))
    cfg.soft_label_negative = float(getattr(cfg, "soft_label_negative", 0.0))
    if "soft_label_positive" in training_cfg and "soft_label_negative" not in training_cfg:
        cfg.soft_label_negative = 1.0 - float(cfg.soft_label_positive)

    cfg.max_clip_value = parse_optional_positive_int(
        cfg.max_clip_value,
        key="training.max_clip_value",
    )
    cfg.normalization_mode = parse_normalization_mode(
        cfg.normalization_mode,
        key="training.normalization_mode",
    )
    cfg.fold_label_foreground_percentile_clip_zscore_stats = None

    if "cv_fold" in training_cfg:
        cfg.cv_fold = normalize_cv_fold(training_cfg["cv_fold"])
    else:
        cfg.cv_fold = normalize_cv_fold(cfg.cv_fold)

    if "train_label_suffix" in training_cfg:
        cfg.train_label_suffix = str(training_cfg["train_label_suffix"])
    if "train_mask_suffix" in training_cfg:
        cfg.train_mask_suffix = str(training_cfg["train_mask_suffix"])
    if "val_label_suffix" in training_cfg:
        cfg.val_label_suffix = str(training_cfg["val_label_suffix"])
    if "val_mask_suffix" in training_cfg:
        cfg.val_mask_suffix = str(training_cfg["val_mask_suffix"])

    if cfg.cv_fold is not None:
        fold_suffix = f"_{cfg.cv_fold}"
        if "train_label_suffix" not in training_cfg:
            cfg.train_label_suffix = fold_suffix
        if "train_mask_suffix" not in training_cfg:
            cfg.train_mask_suffix = fold_suffix
        if "val_label_suffix" not in training_cfg:
            cfg.val_label_suffix = f"_val_{cfg.cv_fold}"
        if "val_mask_suffix" not in training_cfg:
            cfg.val_mask_suffix = f"_val_{cfg.cv_fold}"

    rebuild_augmentations(cfg, augmentation_cfg)
    return cfg
