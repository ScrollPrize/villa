import os
import os.path as osp

os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "109951162777600")

import json
import random
import re
import datetime

import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 109951162777600

import numpy as np
import torch

import albumentations as A
from train_resnet3d_lib import config_metadata_apply as _cfg_meta

def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone = 'resnet3d'
    resnet3d_model_depth = 50
    backbone_pretrained_path = "./r3d50_KM_200ep.pth"
    in_chans = 62  # 65
    encoder_depth = 5
    norm = "batch"  # "batch" | "group"
    group_norm_groups = 32
    # ============== training cfg =============
    size = 256
    tile_size = 256
    stride = tile_size // 8

    train_batch_size = 50  # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = "OneCycleLR"  # "OneCycleLR" | "cosine"
    epochs = 30  # 30

    optimizer = "adamw"  # "adamw" | "sgd"
    adamw_beta2 = 0.999
    adamw_eps = 1e-8
    sgd_momentum = 0.9
    sgd_nesterov = False

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 2e-5
    onecycle_pct_start = 0.15
    onecycle_div_factor = 25.0
    onecycle_final_div_factor = 1e2
    cosine_warmup_pct = 0.15
    # ============== fold =============
    valid_id = None
    stitch_all_val = False
    stitch_downsample = 1
    stitch_train = False
    stitch_train_every_n_epochs = 1
    stitch_use_roi = True
    stitch_log_only_segments = []
    stitch_log_only_every_n_epochs = 10
    stitch_log_only_downsample = 8
    cv_fold = None
    train_label_suffix = ""
    train_mask_suffix = ""
    val_label_suffix = "_val"
    val_mask_suffix = "_val"
    data_backend = "zarr"  # "zarr" (default) | "tiff"
    dataset_root = "train_scrolls"
    dataset_cache_enabled = True
    dataset_cache_check_hash = True
    dataset_cache_dir = "./dataset_cache/train_resnet3d_patch_index"

    # ============== group DRO cfg =============
    objective = "erm"  # "erm" | "group_dro"
    sampler = "shuffle"  # "shuffle" | "group_balanced" | "group_stratified"
    group_stratified_epoch_size_mode = "dataset"  # "dataset" | "min_group"
    loss_mode = "batch"  # "batch" | "per_sample"
    loss_recipe = "dice_bce"  # "dice_bce" | "bce_only"
    bce_smooth_factor = 0.25
    soft_label_positive = 1.0
    soft_label_negative = 0.0
    erm_group_topk = 0  # if >0 and objective=erm+per_sample: optimize mean(worst-k group losses) per batch
    save_every_epoch = False
    save_every_n_epochs = 1
    accumulate_grad_batches = 1

    # ============== eval metrics cfg (validation-only) =============
    # Threshold for confusion-based metrics.
    eval_threshold = 0.5
    # Extra "stitched segment" metrics (expensive, but more faithful for topology/document metrics).
    eval_stitch_metrics = True
    eval_stitch_every_n_epochs = 1
    eval_stitch_every_n_epochs_plus_one = False
    eval_topological_metrics_every_n_epochs = 1
    eval_drd_block_size = 8
    eval_boundary_k = 3
    eval_boundary_tols = [1.0]
    eval_skeleton_thinning_type = "guo_hall"
    eval_enable_skeleton_metrics = True
    eval_component_worst_q = 0.2
    eval_component_worst_k = 2
    eval_component_min_area = 0
    eval_component_pad = 5
    eval_stitch_full_region_metrics = False
    eval_save_stitch_debug_images = True
    eval_save_stitch_debug_images_every_n_epochs = 1
    eval_threshold_grid_min = 0.40
    eval_threshold_grid_max = 0.70
    eval_threshold_grid_steps = 5
    eval_threshold_grid = None
    eval_wandb_media_downsample = 1

    # ============== fixed =============
    pretrained = True

    min_lr = 1e-6
    weight_decay = 1e-6
    max_clip_value = 200
    normalization_mode = "clip_max_div255"
    fold_label_foreground_percentile_clip_zscore_stats = None
    exclude_weight_decay_bias_norm = True
    max_grad_norm = 100

    print_freq = 50
    num_workers = 16
    layer_read_workers = 8

    seed = 130697

    outputs_path = f'./outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'

    # ============== augmentation =============
    train_aug_list = []
    valid_aug_list = []
    rotate = A.Compose([A.Rotate(5, p=1)])
    fourth_augment_p = 0.6
    fourth_augment_min_crop_ratio = 0.9
    fourth_augment_max_crop_ratio = 1.0
    fourth_augment_cutout_max_count = 2
    fourth_augment_cutout_p = 0.6
    invert_augment_p = 0.0


def set_seed(seed=None, cudnn_deterministic=True):
    if seed is None:
        seed = 42

    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = False


def make_dirs(cfg):
    for dir in [cfg.model_dir, cfg.figures_dir, cfg.submission_dir, cfg.log_dir]:
        os.makedirs(dir, exist_ok=True)


def cfg_init(cfg, mode='train'):
    set_seed(cfg.seed)
    # set_env_name()
    # set_dataset_path(cfg)

    if mode == 'train':
        make_dirs(cfg)

def load_metadata_json(path):
    with open(path, "r") as f:
        return json.load(f)


def slugify(value, *, max_len=120):
    value = str(value or "").strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._-")
    if not value:
        value = "run"
    return value[:max_len]


def deep_merge_dict(base, override):
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def unflatten_dict(flat, *, sep="."):
    nested = {}
    for key, value in (flat or {}).items():
        if not isinstance(key, str) or key.startswith("_"):
            continue

        if sep in key:
            cursor = nested
            parts = key.split(sep)
            for part in parts[:-1]:
                if part not in cursor or not isinstance(cursor[part], dict):
                    cursor[part] = {}
                cursor = cursor[part]
            cursor[parts[-1]] = value
        else:
            nested[key] = value
    return nested


def parse_bool_strict(value, *, key):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        parsed_value = value.strip().lower()
        if parsed_value in {"1", "true", "yes", "y"}:
            return True
        if parsed_value in {"0", "false", "no", "n"}:
            return False
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    if isinstance(value, int):
        if value in {0, 1}:
            return bool(value)
        raise ValueError(f"{key} must be a boolean, got {value!r}")
    raise ValueError(f"{key} must be a boolean, got {value!r}")


def resolve_stitch_metadata(merged_config):
    stitch_cfg = dict(merged_config["stitch"])
    training_cfg = dict(merged_config.get("training") or {})
    segment_ids = [str(x).strip() for x in stitch_cfg["segment_ids"]]
    segment_ids = [x for x in segment_ids if x]
    downsample = stitch_cfg.get("downsample")
    if downsample is not None:
        downsample = int(downsample)
        if downsample <= 0:
            raise ValueError(f"metadata_json.stitch.downsample must be > 0, got {downsample}")

    return {
        "segment_ids": segment_ids,
        "mask_suffix": str(stitch_cfg.get("mask_suffix", training_cfg.get("val_mask_suffix", "_val"))),
        "downsample": downsample,
        "schedule": {
            "train_every_n_epochs": stitch_cfg.get("train_every_n_epochs"),
            "eval_every_n_epochs": stitch_cfg.get("eval_every_n_epochs"),
            "eval_plus_one": stitch_cfg.get("eval_plus_one"),
        },
    }


def apply_top_level_stitch_to_cfg(cfg, merged_config):
    stitch_raw = merged_config.get("stitch")
    if stitch_raw is None:
        return

    stitch_cfg = dict(stitch_raw)

    train_every = stitch_cfg.get("train_every_n_epochs")
    eval_every = stitch_cfg.get("eval_every_n_epochs")
    eval_plus_one = stitch_cfg.get("eval_plus_one")
    downsample = stitch_cfg.get("downsample")

    if train_every is not None:
        cfg.stitch_train_every_n_epochs = int(train_every)
    if eval_every is not None:
        cfg.eval_stitch_every_n_epochs = int(eval_every)
    if eval_plus_one is not None:
        cfg.eval_stitch_every_n_epochs_plus_one = parse_bool_strict(
            eval_plus_one,
            key="metadata_json.stitch.eval_plus_one",
        )
    if downsample is not None:
        downsample = int(downsample)
        if downsample <= 0:
            raise ValueError(f"metadata_json.stitch.downsample must be > 0, got {downsample}")
        cfg.stitch_downsample = downsample

    log(
        "stitch schedule "
        f"downsample={int(getattr(cfg, 'stitch_downsample', 1))} "
        f"train_every_n_epochs={int(getattr(cfg, 'stitch_train_every_n_epochs', 1))} "
        f"eval_every_n_epochs={int(getattr(cfg, 'eval_stitch_every_n_epochs', 1))} "
        f"eval_plus_one={bool(getattr(cfg, 'eval_stitch_every_n_epochs_plus_one', False))}"
    )


def validate_stitch_segment_ids(merged_config, segment_ids):
    segments = dict(merged_config.get("segments") or {})
    known_segment_ids = {str(x) for x in segments.keys()}
    missing_segment_ids = [sid for sid in segment_ids if sid not in known_segment_ids]
    if missing_segment_ids:
        raise ValueError(f"stitch segment ids are not defined in metadata_json.segments: {missing_segment_ids!r}")


def parse_optional_positive_int_strict(value, *, key):
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be a positive integer or null, got {value!r}")
    if isinstance(value, int):
        parsed_value = value
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{key} must be a positive integer or null, got {value!r}")
        parsed_value = int(value)
    elif isinstance(value, str):
        parsed_text = value.strip().lower()
        if parsed_text in {"none", "null"}:
            return None
        if not re.fullmatch(r"[+-]?\d+", parsed_text):
            raise ValueError(f"{key} must be a positive integer or null, got {value!r}")
        parsed_value = int(parsed_text)
    else:
        raise ValueError(f"{key} must be a positive integer or null, got {value!r}")
    if parsed_value <= 0:
        raise ValueError(f"{key} must be > 0 when provided, got {parsed_value}")
    return parsed_value


def parse_normalization_mode_strict(value, *, key):
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string, got {type(value).__name__}")
    parsed_value = value.strip().lower()
    valid_modes = {
        "clip_max_div255",
        "train_fold_fg_clip_zscore",
        "train_fold_fg_clip_robust_zscore",
    }
    if parsed_value not in valid_modes:
        raise ValueError(
            f"{key} must be one of {sorted(valid_modes)!r}, got {value!r}"
        )
    return parsed_value


def normalize_cv_fold(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.lower() in {"", "none", "null"}:
            return None
        if stripped.isdigit():
            return int(stripped)
        return stripped
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return value


def normalize_wandb_config(wandb_cfg, *, key_prefix="metadata_json.wandb"):
    if not isinstance(wandb_cfg, dict):
        raise TypeError(f"{key_prefix} must be an object, got {type(wandb_cfg).__name__}")

    enabled = parse_bool_strict(wandb_cfg.get("enabled", False), key=f"{key_prefix}.enabled")
    project = str(wandb_cfg.get("project", "ink-detection")).strip() or "ink-detection"
    entity = str(wandb_cfg.get("entity", "")).strip()
    if enabled and not entity:
        raise ValueError(f"{key_prefix}.entity must be a non-empty string when wandb.enabled=true")

    group = wandb_cfg.get("group")
    if group is None:
        normalized_group = None
    elif isinstance(group, str):
        normalized_group = group.strip() or None
    else:
        normalized_group = str(group).strip() or None

    tags = wandb_cfg.get("tags", [])
    if tags is None:
        tags = []
    if not isinstance(tags, (list, tuple)):
        tags = [tags]
    normalized_tags = []
    for idx, tag in enumerate(tags):
        if not isinstance(tag, str):
            tag = str(tag)
        tag_value = tag.strip()
        if not tag_value:
            continue
        normalized_tags.append(tag_value)

    return {
        "enabled": enabled,
        "project": project,
        "entity": entity,
        "group": normalized_group,
        "tags": normalized_tags,
    }


def resolve_metadata_path(metadata_json, *, base_dir):
    if not metadata_json:
        raise ValueError("--metadata_json is required")
    metadata_path = str(metadata_json)
    if not osp.isabs(metadata_path):
        if not osp.exists(metadata_path):
            metadata_path = osp.join(base_dir, metadata_path)
    return metadata_path


def validate_base_config(base_config):
    if not isinstance(base_config, dict):
        raise TypeError(f"metadata_json root must be an object, got {type(base_config).__name__}")
    required_object_keys = ("training", "group_dro", "wandb", "training_hyperparameters")
    for key in required_object_keys:
        if key not in base_config:
            raise KeyError(f"metadata_json missing required object: {key!r}")
        if not isinstance(base_config[key], dict):
            raise TypeError(
                f"metadata_json.{key} must be an object, got {type(base_config[key]).__name__}"
            )
    return base_config


def load_and_validate_base_config(metadata_json, *, base_dir):
    metadata_path = resolve_metadata_path(metadata_json, base_dir=base_dir)
    log(f"loading metadata_json={metadata_path}")
    base_config = load_metadata_json(metadata_path)
    return validate_base_config(base_config)


def merge_config_with_overrides(base_config, overrides):
    merged_config = json.loads(json.dumps(base_config))
    deep_merge_dict(merged_config, overrides or {})
    if not isinstance(merged_config.get("wandb"), dict):
        merged_config["wandb"] = {}
    merged_config["wandb"] = normalize_wandb_config(merged_config["wandb"], key_prefix="merged_config.wandb")
    return merged_config


def apply_metadata_hyperparameters(cfg, metadata):
    return _cfg_meta.apply_metadata_hyperparameters(
        cfg,
        metadata,
        parse_bool=parse_bool_strict,
        parse_optional_positive_int=parse_optional_positive_int_strict,
        parse_normalization_mode=parse_normalization_mode_strict,
        normalize_cv_fold=normalize_cv_fold,
    )
