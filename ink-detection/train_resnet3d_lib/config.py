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
from albumentations.pytorch import ToTensorV2


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'./'

    exp_name = 'pretraining_all'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone = 'resnet3d'
    in_chans = 30  # 65
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
    valid_id = '20230820203112'
    stitch_all_val = False
    stitch_downsample = 1
    stitch_train = False
    stitch_train_every_n_epochs = 1
    cv_fold = None
    train_label_suffix = ""
    train_mask_suffix = ""
    val_label_suffix = "_val"
    val_mask_suffix = "_val"

    # ============== group DRO cfg =============
    objective = "erm"  # "erm" | "group_dro"
    sampler = "shuffle"  # "shuffle" | "group_balanced" | "group_stratified"
    loss_mode = "batch"  # "batch" | "per_sample"
    erm_group_topk = 0  # if >0 and objective=erm+per_sample: optimize mean(worst-k group losses) per batch
    save_every_epoch = False
    accumulate_grad_batches = 1

    # objective_cv = 'binary'  # 'binary', 'multiclass', 'regression'
    metric_direction = 'maximize'  # maximize, 'minimize'
    # metrics = 'dice_coef'

    # ============== fixed =============
    pretrained = True
    inf_weight = 'best'  # 'best'

    min_lr = 1e-6
    weight_decay = 1e-6
    exclude_weight_decay_bias_norm = True
    max_grad_norm = 100

    print_freq = 50
    num_workers = 16
    layer_read_workers = 8

    seed = 130697

    # ============== set dataset path =============
    print('set dataset path')

    outputs_path = f'./outputs/{comp_name}/{exp_name}/'

    submission_dir = outputs_path + 'submissions/'
    submission_path = submission_dir + f'submission_{exp_name}.csv'

    model_dir = outputs_path + \
        f'{comp_name}-models/'

    figures_dir = outputs_path + 'figures/'

    log_dir = outputs_path + 'logs/'
    log_path = log_dir + f'{exp_name}.txt'

    # ============== augmentation =============
    train_aug_list = []
    valid_aug_list = []
    rotate = A.Compose([A.Rotate(5, p=1)])


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

DEFAULT_FRAGMENT_IDS = [
    '20231210121321', '20231106155350', '20231005123336', '20230820203112', '20230620230619',
    '20230826170124', '20230702185753', '20230522215721', '20230531193658', '20230520175435',
    '20230903193206', '20230902141231', '20231007101615', '20230929220924', 'recto', 'verso',
    '20231016151000', '20231012184423', '20231031143850'
]

REVERSE_LAYER_FRAGMENT_IDS_FALLBACK = {
    '20230701020044', 'verso', '20230901184804', '20230901234823', '20230531193658', '20231007101615',
    '20231005123333', '20231011144857', '20230522215721', '20230919113918', '20230625171244',
    '20231022170900', '20231012173610', '20231016151000'
}


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

    size = cfg.size
    in_chans = cfg.in_chans

    hflip_p = float(augmentation_cfg.get("horizontal_flip", 0.5))
    vflip_p = float(augmentation_cfg.get("vertical_flip", 0.5))
    shift_scale_rotate = augmentation_cfg.get("shift_scale_rotate", {})
    blur_cfg = augmentation_cfg.get("blur", {})
    coarse_dropout_cfg = augmentation_cfg.get("coarse_dropout", {})

    blur_types = set(blur_cfg.get("types", ["GaussianBlur", "MotionBlur"]))
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
    hp = metadata.get("training_hyperparameters", {})
    model_hp = hp.get("model", {})
    train_hp = hp.get("training", {})
    training_cfg = metadata.get("training", {}) or {}

    for k in ["model_name", "backbone", "encoder_depth", "target_size", "in_chans", "norm", "group_norm_groups"]:
        if k in model_hp:
            if k == "norm":
                setattr(cfg, "norm", str(model_hp[k]).lower())
            elif k == "group_norm_groups":
                setattr(cfg, "group_norm_groups", int(model_hp[k]))
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
        ("pretrained", "pretrained"),
        ("num_workers", "num_workers"),
        ("layer_read_workers", "layer_read_workers"),
        ("seed", "seed"),
    ]:
        if k in train_hp:
            setattr(cfg, attr, train_hp[k])

    if getattr(cfg, "stride", None) is None:
        cfg.stride = cfg.tile_size // 8

    cfg.objective = str(training_cfg.get("objective", getattr(cfg, "objective", "erm"))).lower()
    cfg.sampler = str(training_cfg.get("sampler", getattr(cfg, "sampler", "shuffle"))).lower()
    cfg.loss_mode = str(training_cfg.get("loss_mode", getattr(cfg, "loss_mode", "batch"))).lower()
    cfg.erm_group_topk = int(training_cfg.get("erm_group_topk", getattr(cfg, "erm_group_topk", 0) or 0))
    cfg.save_every_epoch = bool(training_cfg.get("save_every_epoch", getattr(cfg, "save_every_epoch", False)))
    cfg.stitch_all_val = bool(training_cfg.get("stitch_all_val", getattr(cfg, "stitch_all_val", False)))
    cfg.stitch_train = bool(training_cfg.get("stitch_train", getattr(cfg, "stitch_train", False)))
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

    rebuild_augmentations(cfg, hp.get("augmentation"))
    return cfg
