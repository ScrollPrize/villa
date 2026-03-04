from train_resnet3d_lib.data.augmentations import rebuild_augmentations


MODEL_SCALARS = {
    "model_name": str,
    "backbone": str,
    "backbone_pretrained_path": str,
    "resnet3d_model_depth": int,
    "encoder_depth": int,
    "target_size": int,
    "in_chans": int,
    "norm": str,
    "group_norm_groups": int,
}

TRAIN_HPARAM_SCALARS = {
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
}

TRAIN_HPARAM_BOOLS = {
    "use_amp",
    "sgd_nesterov",
    "exclude_weight_decay_bias_norm",
    "eval_stitch_metrics",
    "eval_enable_skeleton_metrics",
    "eval_stitch_full_region_metrics",
    "eval_save_stitch_debug_images",
}

TRAIN_CFG_SCALARS = {
    "erm_group_topk": int,
    "save_every_n_epochs": int,
    "data_backend": str,
    "dataset_root": str,
    "dataset_cache_dir": str,
    "stitch_log_only_segments": list,
    "stitch_log_only_every_n_epochs": int,
    "stitch_log_only_downsample": int,
}

TRAIN_CFG_BOOLS = {
    "save_every_epoch",
    "stitch_all_val",
    "stitch_train",
    "pretrained",
    "dataset_cache_enabled",
    "dataset_cache_check_hash",
    "stitch_use_roi",
}

TRAIN_CFG_LOWER = {
    "objective",
    "sampler",
    "group_stratified_epoch_size_mode",
    "loss_mode",
    "loss_recipe",
}

VALID_OBJECTIVES = {"erm", "group_dro"}
VALID_SAMPLERS = {"shuffle", "group_balanced", "group_stratified"}
VALID_LOSS_MODES = {"batch", "per_sample"}
VALID_LOSS_RECIPES = {"dice_bce", "bce_only"}
VALID_DATA_BACKENDS = {"zarr", "tiff"}


def _section(parent, key, *, key_prefix):
    value = parent.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"{key_prefix}.{key} must be an object, got {type(value).__name__}")
    return value


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
        parsed = parse_bool(source[key], key=f"{key_prefix}.{key}")
        if not isinstance(parsed, bool):
            raise TypeError(
                f"{key_prefix}.{key} parser must return bool, got {type(parsed).__name__}"
            )
        setattr(cfg, key, parsed)


def _ensure_enum(value, *, key, allowed):
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)!r}, got {value!r}")


def _ensure_positive_int(value, *, key):
    ivalue = int(value)
    if ivalue <= 0:
        raise ValueError(f"{key} must be > 0, got {ivalue}")


def _ensure_non_negative_int(value, *, key):
    ivalue = int(value)
    if ivalue < 0:
        raise ValueError(f"{key} must be >= 0, got {ivalue}")


def _ensure_number_list(value, *, key):
    if not isinstance(value, list):
        raise ValueError(f"{key} must be a list")
    for idx, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{key}[{idx}] must be numeric, got {type(item).__name__}")


def _ensure_float_range(value, *, key, min_value, max_value):
    fvalue = float(value)
    if fvalue < float(min_value) or fvalue > float(max_value):
        raise ValueError(f"{key} must be in [{min_value}, {max_value}], got {fvalue}")
    return fvalue


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

    hp = _section(metadata, "training_hyperparameters", key_prefix="metadata")
    model_hp = _section(hp, "model", key_prefix="metadata.training_hyperparameters")
    train_hp = _section(hp, "training", key_prefix="metadata.training_hyperparameters")
    augmentation_hp = _section(hp, "augmentation", key_prefix="metadata.training_hyperparameters")
    training_cfg = _section(metadata, "training", key_prefix="metadata")

    _apply_scalar_fields(cfg, model_hp, casts=MODEL_SCALARS)
    _apply_scalar_fields(cfg, train_hp, casts=TRAIN_HPARAM_SCALARS)
    _apply_bool_fields(
        cfg,
        train_hp,
        keys=TRAIN_HPARAM_BOOLS,
        parse_bool=parse_bool,
        key_prefix="training_hyperparameters.training",
    )

    for key in TRAIN_CFG_LOWER:
        if key not in training_cfg:
            continue
        setattr(cfg, key, str(training_cfg[key]).strip().lower())
    _apply_scalar_fields(cfg, training_cfg, casts=TRAIN_CFG_SCALARS)
    _apply_bool_fields(
        cfg,
        training_cfg,
        keys=TRAIN_CFG_BOOLS,
        parse_bool=parse_bool,
        key_prefix="training",
    )

    if "train_batch_size" in train_hp and "valid_batch_size" not in train_hp:
        cfg.valid_batch_size = int(cfg.train_batch_size)

    cfg.norm = str(cfg.norm).strip().lower()
    cfg.optimizer = str(cfg.optimizer).strip().lower()
    cfg.data_backend = str(cfg.data_backend).strip().lower()
    cfg.loss_recipe = str(getattr(cfg, "loss_recipe", "dice_bce")).strip().lower()
    cfg.bce_smooth_factor = float(getattr(cfg, "bce_smooth_factor", 0.25))
    cfg.soft_label_positive = float(getattr(cfg, "soft_label_positive", 1.0))
    cfg.soft_label_negative = float(getattr(cfg, "soft_label_negative", 0.0))
    # Allow sweeps to vary only soft_label_positive while keeping pairs complementary.
    if "soft_label_positive" in train_hp and "soft_label_negative" not in train_hp:
        cfg.soft_label_negative = 1.0 - float(cfg.soft_label_positive)
    _ensure_enum(cfg.objective, key="training.objective", allowed=VALID_OBJECTIVES)
    _ensure_enum(cfg.sampler, key="training.sampler", allowed=VALID_SAMPLERS)
    _ensure_enum(cfg.loss_mode, key="training.loss_mode", allowed=VALID_LOSS_MODES)
    _ensure_enum(cfg.loss_recipe, key="training.loss_recipe", allowed=VALID_LOSS_RECIPES)
    _ensure_enum(cfg.data_backend, key="training.data_backend", allowed=VALID_DATA_BACKENDS)
    _ensure_positive_int(cfg.train_batch_size, key="training_hyperparameters.training.train_batch_size")
    _ensure_positive_int(cfg.valid_batch_size, key="training_hyperparameters.training.valid_batch_size")
    _ensure_positive_int(cfg.epochs, key="training_hyperparameters.training.epochs")
    _ensure_positive_int(cfg.stitch_downsample, key="stitch.downsample")
    _ensure_non_negative_int(cfg.num_workers, key="training_hyperparameters.training.num_workers")
    _ensure_non_negative_int(cfg.layer_read_workers, key="training_hyperparameters.training.layer_read_workers")
    boundary_tols = getattr(cfg, "eval_boundary_tols", None)
    if boundary_tols is not None:
        _ensure_number_list(boundary_tols, key="training_hyperparameters.training.eval_boundary_tols")
    stitch_log_only_segments = getattr(cfg, "stitch_log_only_segments", [])
    if not isinstance(stitch_log_only_segments, list):
        raise ValueError("training.stitch_log_only_segments must be a list")

    cfg.bce_smooth_factor = _ensure_float_range(
        cfg.bce_smooth_factor,
        key="training_hyperparameters.training.bce_smooth_factor",
        min_value=0.0,
        max_value=0.5,
    )
    cfg.soft_label_positive = _ensure_float_range(
        cfg.soft_label_positive,
        key="training_hyperparameters.training.soft_label_positive",
        min_value=0.0,
        max_value=1.0,
    )
    cfg.soft_label_negative = _ensure_float_range(
        cfg.soft_label_negative,
        key="training_hyperparameters.training.soft_label_negative",
        min_value=0.0,
        max_value=1.0,
    )
    if cfg.soft_label_positive <= cfg.soft_label_negative:
        raise ValueError(
            "training_hyperparameters.training.soft_label_positive must be greater than "
            f"soft_label_negative, got {cfg.soft_label_positive} <= {cfg.soft_label_negative}"
        )

    cfg.max_clip_value = parse_optional_positive_int(
        cfg.max_clip_value,
        key="training_hyperparameters.training.max_clip_value",
    )
    cfg.normalization_mode = parse_normalization_mode(
        cfg.normalization_mode,
        key="training_hyperparameters.training.normalization_mode",
    )
    cfg.fold_label_foreground_percentile_clip_zscore_stats = None
    if cfg.normalization_mode in {
        "train_fold_fg_clip_zscore",
        "train_fold_fg_clip_robust_zscore",
    } and cfg.max_clip_value is not None:
        raise ValueError(
            "training_hyperparameters.training.max_clip_value must be null when "
            "using train_fold_fg_clip_zscore/train_fold_fg_clip_robust_zscore"
        )

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

    rebuild_augmentations(cfg, augmentation_hp)
    return cfg
