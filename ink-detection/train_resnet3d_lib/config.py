import os

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
from albumentations.pytorch import ToTensorV2


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

    # ============== group DRO cfg =============
    objective = "erm"  # "erm" | "group_dro"
    sampler = "shuffle"  # "shuffle" | "group_balanced" | "group_stratified"
    loss_mode = "batch"  # "batch" | "per_sample"
    erm_group_topk = 0  # if >0 and objective=erm+per_sample: optimize mean(worst-k group losses) per batch
    save_every_epoch = False
    accumulate_grad_batches = 1

    # ============== eval metrics cfg (validation-only) =============
    # Threshold for confusion-based metrics.
    eval_threshold = 0.5
    # Extra "stitched segment" metrics (expensive, but more faithful for topology/document metrics).
    eval_stitch_metrics = True
    eval_stitch_every_n_epochs = 1
    eval_drd_block_size = 8
    eval_boundary_k = 3
    eval_boundary_tols = [1.0]
    eval_skeleton_radius = [1]
    eval_component_worst_q = 0.2
    eval_component_worst_k = 2
    eval_component_min_area = 0
    eval_component_pad = 5
    eval_skeleton_thinning_type = "guo_hall"
    eval_stitch_full_region_metrics = False
    eval_save_stitched_inputs = True
    eval_save_component_debug_images = False
    eval_component_debug_max_items = 24
    eval_threshold_grid_min = 0.40
    eval_threshold_grid_max = 0.70
    eval_threshold_grid_steps = 5
    eval_threshold_grid = None
    eval_wandb_media_downsample = 1

    # ============== fixed =============
    pretrained = True

    min_lr = 1e-6
    weight_decay = 1e-6
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


def rebuild_augmentations(cfg, augmentation_cfg=None):
    if augmentation_cfg is None:
        augmentation_cfg = {}
    if not isinstance(augmentation_cfg, dict):
        raise TypeError(
            f"metadata.training_hyperparameters.augmentation must be an object, got {type(augmentation_cfg).__name__}"
        )
    if "fourth_augment" not in augmentation_cfg or not isinstance(augmentation_cfg["fourth_augment"], dict):
        raise KeyError(
            "metadata.training_hyperparameters.augmentation missing required object: 'fourth_augment'"
        )

    size = cfg.size
    in_chans = cfg.in_chans
    fourth_augment_cfg = augmentation_cfg["fourth_augment"]
    required_fourth_augment_keys = (
        "p",
        "min_crop_ratio",
        "max_crop_ratio",
        "cutout_max_count",
        "cutout_p",
    )
    missing_fourth_augment_keys = [k for k in required_fourth_augment_keys if k not in fourth_augment_cfg]
    if missing_fourth_augment_keys:
        raise KeyError(
            "metadata.training_hyperparameters.augmentation.fourth_augment missing required keys: "
            f"{missing_fourth_augment_keys}"
        )

    cfg.fourth_augment_p = float(fourth_augment_cfg["p"])
    cfg.fourth_augment_min_crop_ratio = float(fourth_augment_cfg["min_crop_ratio"])
    cfg.fourth_augment_max_crop_ratio = float(fourth_augment_cfg["max_crop_ratio"])
    cfg.fourth_augment_cutout_max_count = int(fourth_augment_cfg["cutout_max_count"])
    cfg.fourth_augment_cutout_p = float(fourth_augment_cfg["cutout_p"])

    if not (0.0 <= cfg.fourth_augment_p <= 1.0):
        raise ValueError(f"augmentation.fourth_augment.p must be in [0,1], got {cfg.fourth_augment_p}")
    if not (0.0 < cfg.fourth_augment_min_crop_ratio <= cfg.fourth_augment_max_crop_ratio <= 1.0):
        raise ValueError(
            "augmentation.fourth_augment ratios must satisfy 0 < min_crop_ratio <= max_crop_ratio <= 1, "
            f"got min={cfg.fourth_augment_min_crop_ratio}, max={cfg.fourth_augment_max_crop_ratio}"
        )
    if cfg.fourth_augment_cutout_max_count < 0:
        raise ValueError(
            "augmentation.fourth_augment.cutout_max_count must be >= 0, "
            f"got {cfg.fourth_augment_cutout_max_count}"
        )
    if not (0.0 <= cfg.fourth_augment_cutout_p <= 1.0):
        raise ValueError(
            f"augmentation.fourth_augment.cutout_p must be in [0,1], got {cfg.fourth_augment_cutout_p}"
        )

    hflip_p = float(augmentation_cfg.get("horizontal_flip", 0.5))
    vflip_p = float(augmentation_cfg.get("vertical_flip", 0.5))
    shift_scale_rotate = augmentation_cfg.get("shift_scale_rotate", {})
    blur_cfg = augmentation_cfg.get("blur", {})
    coarse_dropout_cfg = augmentation_cfg.get("coarse_dropout", {})

    blur_types = set(blur_cfg.get("types", ["GaussNoise", "GaussianBlur", "MotionBlur"]))
    blur_transforms = []
    if "GaussNoise" in blur_types:
        gauss_noise_std_range = (
            float(np.sqrt(10.0) / 255.0),
            float(np.sqrt(50.0) / 255.0),
        )
        try:
            blur_transforms.append(A.GaussNoise(std_range=gauss_noise_std_range))
        except TypeError:
            blur_transforms.append(A.GaussNoise(var_limit=(10, 50)))
    if "GaussianBlur" in blur_types:
        blur_transforms.append(A.GaussianBlur())
    if "MotionBlur" in blur_types:
        blur_transforms.append(A.MotionBlur())

    coarse_dropout_p = float(coarse_dropout_cfg.get("p", 0.5))
    max_holes = int(coarse_dropout_cfg.get("max_holes", 2))
    max_width = int(size * float(coarse_dropout_cfg.get("max_width_ratio", 0.2)))
    max_height = int(size * float(coarse_dropout_cfg.get("max_height_ratio", 0.2)))
    try:
        coarse_dropout = A.CoarseDropout(
            num_holes_range=(1, max_holes),
            hole_height_range=(1, max_height),
            hole_width_range=(1, max_width),
            fill=0,
            fill_mask=0,
            p=coarse_dropout_p,
        )
    except TypeError:
        coarse_dropout = A.CoarseDropout(
            max_holes=max_holes,
            max_width=max_width,
            max_height=max_height,
            mask_fill_value=0,
            p=coarse_dropout_p,
        )

    cfg.train_aug_list = [
        A.Resize(size, size),
        A.HorizontalFlip(p=hflip_p),
        A.VerticalFlip(p=vflip_p),
        A.RandomBrightnessContrast(p=0.75),
        A.ShiftScaleRotate(
            rotate_limit=int(shift_scale_rotate.get("rotate_limit", 360)),
            shift_limit=float(shift_scale_rotate.get("shift_limit", 0.15)),
            scale_limit=float(shift_scale_rotate.get("scale_limit", 0.1)),
            p=float(shift_scale_rotate.get("p", 0.75)),
        ),
        A.OneOf(blur_transforms or [A.GaussianBlur(), A.MotionBlur()], p=float(blur_cfg.get("p", 0.4))),
        coarse_dropout,
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    cfg.valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]


def apply_metadata_hyperparameters(cfg, metadata):
    if not isinstance(metadata, dict):
        raise TypeError(f"metadata must be an object, got {type(metadata).__name__}")
    if "training_hyperparameters" not in metadata or not isinstance(metadata["training_hyperparameters"], dict):
        raise KeyError("metadata missing required object: 'training_hyperparameters'")
    if "training" not in metadata or not isinstance(metadata["training"], dict):
        raise KeyError("metadata missing required object: 'training'")

    hp = metadata["training_hyperparameters"]
    if "model" not in hp or not isinstance(hp["model"], dict):
        raise KeyError("metadata.training_hyperparameters missing required object: 'model'")
    if "training" not in hp or not isinstance(hp["training"], dict):
        raise KeyError("metadata.training_hyperparameters missing required object: 'training'")
    if "augmentation" not in hp or not isinstance(hp["augmentation"], dict):
        raise KeyError("metadata.training_hyperparameters missing required object: 'augmentation'")
    model_hp = hp["model"]
    train_hp = hp["training"]
    training_cfg = metadata["training"]
    required_model_keys = ["model_name", "backbone", "encoder_depth", "target_size", "in_chans", "norm", "group_norm_groups"]
    missing_model_keys = [k for k in required_model_keys if k not in model_hp]
    if missing_model_keys:
        raise KeyError(f"metadata.training_hyperparameters.model missing required keys: {missing_model_keys}")

    required_training_keys = [
        "size",
        "tile_size",
        "stride",
        "train_batch_size",
        "valid_batch_size",
        "use_amp",
        "epochs",
        "scheduler",
        "lr",
        "min_lr",
        "weight_decay",
        "num_workers",
        "layer_read_workers",
        "seed",
    ]
    missing_training_keys = [k for k in required_training_keys if k not in train_hp]
    if missing_training_keys:
        raise KeyError(f"metadata.training_hyperparameters.training missing required keys: {missing_training_keys}")

    for k in required_model_keys:
        if k == "norm":
            setattr(cfg, "norm", str(model_hp[k]).lower())
        elif k == "group_norm_groups":
            setattr(cfg, "group_norm_groups", int(model_hp[k]))
        elif k in {"encoder_depth", "target_size", "in_chans"}:
            setattr(cfg, k, int(model_hp[k]))
        else:
            setattr(cfg, k, model_hp[k])

    for k, attr in [
        ("size", "size"),
        ("tile_size", "tile_size"),
        ("stride", "stride"),
        ("train_batch_size", "train_batch_size"),
        ("valid_batch_size", "valid_batch_size"),
        ("use_amp", "use_amp"),
        ("accumulate_grad_batches", "accumulate_grad_batches"),
        ("epochs", "epochs"),
        ("scheduler", "scheduler"),
        ("warmup_factor", "warmup_factor"),
        ("lr", "lr"),
        ("onecycle_pct_start", "onecycle_pct_start"),
        ("onecycle_div_factor", "onecycle_div_factor"),
        ("onecycle_final_div_factor", "onecycle_final_div_factor"),
        ("cosine_warmup_pct", "cosine_warmup_pct"),
        ("min_lr", "min_lr"),
        ("weight_decay", "weight_decay"),
        ("exclude_weight_decay_bias_norm", "exclude_weight_decay_bias_norm"),
        ("max_grad_norm", "max_grad_norm"),
        ("eval_threshold", "eval_threshold"),
        ("eval_stitch_metrics", "eval_stitch_metrics"),
        ("eval_stitch_every_n_epochs", "eval_stitch_every_n_epochs"),
        ("eval_drd_block_size", "eval_drd_block_size"),
        ("eval_boundary_k", "eval_boundary_k"),
        ("eval_boundary_tols", "eval_boundary_tols"),
        ("eval_skeleton_radius", "eval_skeleton_radius"),
        ("eval_component_worst_q", "eval_component_worst_q"),
        ("eval_component_worst_k", "eval_component_worst_k"),
        ("eval_component_min_area", "eval_component_min_area"),
        ("eval_component_pad", "eval_component_pad"),
        ("eval_skeleton_thinning_type", "eval_skeleton_thinning_type"),
        ("eval_stitch_full_region_metrics", "eval_stitch_full_region_metrics"),
        ("eval_save_stitched_inputs", "eval_save_stitched_inputs"),
        ("eval_save_component_debug_images", "eval_save_component_debug_images"),
        ("eval_component_debug_max_items", "eval_component_debug_max_items"),
        ("eval_threshold_grid_min", "eval_threshold_grid_min"),
        ("eval_threshold_grid_max", "eval_threshold_grid_max"),
        ("eval_threshold_grid_steps", "eval_threshold_grid_steps"),
        ("eval_threshold_grid", "eval_threshold_grid"),
        ("eval_wandb_media_downsample", "eval_wandb_media_downsample"),
        ("pretrained", "pretrained"),
        ("num_workers", "num_workers"),
        ("layer_read_workers", "layer_read_workers"),
        ("seed", "seed"),
        ("dataset_root", "dataset_root"),
    ]:
        if k in train_hp:
            setattr(cfg, attr, train_hp[k])

    numeric_constraints = [
        ("size", "training_hyperparameters.training.size", 1),
        ("tile_size", "training_hyperparameters.training.tile_size", 1),
        ("stride", "training_hyperparameters.training.stride", 1),
        ("in_chans", "training_hyperparameters.model.in_chans", 1),
        ("train_batch_size", "training_hyperparameters.training.train_batch_size", 1),
        ("valid_batch_size", "training_hyperparameters.training.valid_batch_size", 1),
        ("epochs", "training_hyperparameters.training.epochs", 1),
        ("num_workers", "training_hyperparameters.training.num_workers", 0),
        ("layer_read_workers", "training_hyperparameters.training.layer_read_workers", 1),
        ("group_norm_groups", "training_hyperparameters.model.group_norm_groups", 1),
    ]
    for attr, meta_key, min_value in numeric_constraints:
        value = int(getattr(cfg, attr))
        setattr(cfg, attr, value)
        if value < min_value:
            constraint = ">= 0" if min_value == 0 else "> 0"
            raise ValueError(f"{meta_key} must be {constraint}, got {value}")
    cfg.seed = int(cfg.seed)

    if cfg.norm not in {"batch", "group"}:
        raise ValueError(f"training_hyperparameters.model.norm must be 'batch' or 'group', got {cfg.norm!r}")
    cfg.eval_stitch_every_n_epochs = int(getattr(cfg, "eval_stitch_every_n_epochs", 1))
    if cfg.eval_stitch_every_n_epochs < 1:
        raise ValueError(
            "training_hyperparameters.training.eval_stitch_every_n_epochs must be >= 1, "
            f"got {cfg.eval_stitch_every_n_epochs}"
        )
    cfg.eval_wandb_media_downsample = int(getattr(cfg, "eval_wandb_media_downsample", 1))
    if cfg.eval_wandb_media_downsample < 1:
        raise ValueError(
            "training_hyperparameters.training.eval_wandb_media_downsample must be >= 1, "
            f"got {cfg.eval_wandb_media_downsample}"
        )
    cfg.eval_component_min_area = int(getattr(cfg, "eval_component_min_area", 0) or 0)
    if cfg.eval_component_min_area < 0:
        raise ValueError(
            "training_hyperparameters.training.eval_component_min_area must be >= 0, "
            f"got {cfg.eval_component_min_area}"
        )
    cfg.eval_skeleton_thinning_type = str(getattr(cfg, "eval_skeleton_thinning_type", "zhang_suen")).strip().lower()
    if cfg.eval_skeleton_thinning_type not in {"zhang_suen", "guo_hall"}:
        raise ValueError(
            "training_hyperparameters.training.eval_skeleton_thinning_type must be "
            f"'zhang_suen' or 'guo_hall', got {cfg.eval_skeleton_thinning_type!r}"
        )

    cfg.objective = str(training_cfg.get("objective", getattr(cfg, "objective", "erm"))).lower()
    cfg.sampler = str(training_cfg.get("sampler", getattr(cfg, "sampler", "shuffle"))).lower()
    cfg.loss_mode = str(training_cfg.get("loss_mode", getattr(cfg, "loss_mode", "batch"))).lower()
    cfg.erm_group_topk = int(training_cfg.get("erm_group_topk", getattr(cfg, "erm_group_topk", 0) or 0))
    cfg.save_every_epoch = bool(training_cfg.get("save_every_epoch", getattr(cfg, "save_every_epoch", False)))
    cfg.stitch_all_val = bool(training_cfg.get("stitch_all_val", getattr(cfg, "stitch_all_val", False)))
    cfg.stitch_train = bool(training_cfg.get("stitch_train", getattr(cfg, "stitch_train", False)))
    cfg.data_backend = str(training_cfg.get("data_backend", getattr(cfg, "data_backend", "zarr"))).lower().strip()
    if cfg.data_backend not in {"zarr", "tiff"}:
        raise ValueError(f"training.data_backend must be 'zarr' or 'tiff', got {cfg.data_backend!r}")
    cfg.dataset_root = str(training_cfg.get("dataset_root", getattr(cfg, "dataset_root", "train_scrolls")))
    cfg.stitch_log_only_segments = list(
        training_cfg.get("stitch_log_only_segments", getattr(cfg, "stitch_log_only_segments", [])) or []
    )
    cfg.stitch_log_only_every_n_epochs = int(
        training_cfg.get(
            "stitch_log_only_every_n_epochs",
            getattr(cfg, "stitch_log_only_every_n_epochs", 10) or 10,
        )
    )
    cfg.stitch_log_only_every_n_epochs = max(1, int(cfg.stitch_log_only_every_n_epochs))
    cfg.stitch_log_only_downsample = int(
        training_cfg.get(
            "stitch_log_only_downsample",
            getattr(cfg, "stitch_log_only_downsample", getattr(cfg, "stitch_downsample", 1)),
        )
        or 1
    )
    cfg.stitch_log_only_downsample = max(1, int(cfg.stitch_log_only_downsample))
    cfg.stitch_use_roi = bool(training_cfg.get("stitch_use_roi", getattr(cfg, "stitch_use_roi", True)))
    cfg.stitch_train_every_n_epochs = int(
        training_cfg.get("stitch_train_every_n_epochs", getattr(cfg, "stitch_train_every_n_epochs", 1) or 1)
    )
    cfg.stitch_train_every_n_epochs = max(1, int(cfg.stitch_train_every_n_epochs))
    if "stitch_downsample" in training_cfg:
        cfg.stitch_downsample = int(training_cfg["stitch_downsample"])
    else:
        cfg.stitch_downsample = 8 if cfg.stitch_all_val else int(getattr(cfg, "stitch_downsample", 1))
    cfg.stitch_downsample = max(1, int(cfg.stitch_downsample))

    cv_fold = training_cfg.get("cv_fold", getattr(cfg, "cv_fold", None))
    if isinstance(cv_fold, str) and cv_fold.strip().lower() in {"", "none", "null"}:
        cv_fold = None
    if isinstance(cv_fold, str):
        cv_fold = cv_fold.strip()
        if cv_fold.isdigit():
            cv_fold = int(cv_fold)
    if isinstance(cv_fold, float) and float(cv_fold).is_integer():
        cv_fold = int(cv_fold)
    cfg.cv_fold = cv_fold

    def _suffix_or_default(value, default):
        if value is None:
            return default
        return str(value)

    cfg.train_label_suffix = _suffix_or_default(
        training_cfg.get("train_label_suffix", getattr(cfg, "train_label_suffix", "")),
        "",
    )
    cfg.train_mask_suffix = _suffix_or_default(
        training_cfg.get("train_mask_suffix", getattr(cfg, "train_mask_suffix", "")),
        "",
    )
    cfg.val_label_suffix = _suffix_or_default(
        training_cfg.get("val_label_suffix", getattr(cfg, "val_label_suffix", "_val")),
        "_val",
    )
    cfg.val_mask_suffix = _suffix_or_default(
        training_cfg.get("val_mask_suffix", getattr(cfg, "val_mask_suffix", "_val")),
        "_val",
    )

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

    rebuild_augmentations(cfg, hp["augmentation"])
    return cfg
