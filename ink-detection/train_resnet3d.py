import os.path as osp
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "109951162777600")
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import random
import yaml

import numpy as np
import pandas as pd

import wandb

from torch.utils.data import DataLoader

import pandas as pd
import os
import random
from contextlib import contextmanager
import cv2

import scipy as sp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import argparse
import inspect
import torch
import torch.nn as nn
from torch.optim import AdamW

import datetime
import uuid
import re
import time
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from group_dro import GroupDROComputer
from samplers import GroupStratifiedBatchSampler
from models.i3dallnl import InceptionI3d
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
from models.resnetall import generate_model
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 109951162777600


def log(msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _read_gray(path):
    if not osp.exists(path):
        return None

    # OpenCV can emit noisy libpng warnings (e.g. "chunk data is too large") for some PNGs.
    # Prefer PIL for PNGs to keep logs clean; fall back to OpenCV if PIL fails.
    if path.lower().endswith(".png"):
        try:
            with PIL.Image.open(path) as im:
                return np.array(im.convert("L"))
        except Exception:
            pass

    try:
        img = cv2.imread(path, 0)
        if img is not None:
            return img
    except cv2.error:
        pass

    try:
        with PIL.Image.open(path) as im:
            return np.array(im.convert("L"))
    except Exception as e:
        raise RuntimeError(f"Could not read image: {path}. Error: {e}") from e
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
    backbone='resnet3d'
    in_chans = 30 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 256
    tile_size = 256
    stride = tile_size // 8

    train_batch_size = 50 # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 2e-5
    onecycle_pct_start = 0.15
    onecycle_div_factor = 25.0
    onecycle_final_div_factor = 1e2
    # ============== fold =============
    valid_id = '20230820203112'
    stitch_all_val = False
    stitch_downsample = 1

    # ============== group DRO cfg =============
    objective = "erm"  # "erm" | "group_dro"
    sampler = "shuffle"  # "shuffle" | "group_balanced" | "group_stratified"
    loss_mode = "batch"  # "batch" | "per_sample"
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
    rotate = A.Compose([A.Rotate(5,p=1)])
def init_logger(log_file):
    from logging import getLogger, INFO, FileHandler, Formatter, StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_FRAGMENT_IDS = [
    '20231210121321','20231106155350','20231005123336','20230820203112','20230620230619',
    '20230826170124','20230702185753','20230522215721','20230531193658','20230520175435',
    '20230903193206','20230902141231','20231007101615','20230929220924','recto','verso',
    '20231016151000','20231012184423','20231031143850'
]

REVERSE_LAYER_FRAGMENT_IDS_FALLBACK = {
    '20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615',
    '20231005123333','20231011144857','20230522215721','20230919113918','20230625171244',
    '20231022170900','20231012173610','20231016151000'
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

    for k in ["model_name", "backbone", "encoder_depth", "target_size", "in_chans"]:
        if k in model_hp:
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
        ("min_lr", "min_lr"),
        ("weight_decay", "weight_decay"),
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
    cfg.save_every_epoch = bool(training_cfg.get("save_every_epoch", getattr(cfg, "save_every_epoch", False)))
    cfg.stitch_all_val = bool(training_cfg.get("stitch_all_val", getattr(cfg, "stitch_all_val", False)))
    if "stitch_downsample" in training_cfg:
        cfg.stitch_downsample = int(training_cfg["stitch_downsample"])
    else:
        cfg.stitch_downsample = 8 if cfg.stitch_all_val else int(getattr(cfg, "stitch_downsample", 1))
    cfg.stitch_downsample = max(1, int(cfg.stitch_downsample))

    rebuild_augmentations(cfg, hp.get("augmentation"))
    return cfg


def read_image_layers(
    fragment_id,
    start_idx=15,
    end_idx=45,
    *,
    layer_range=None,
):
    layers_dir = osp.join("train_scrolls", fragment_id, "layers")
    layer_exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

    def _iter_layer_paths(layer_idx):
        # Support 00.tif, 000.tif, 0000.tif, etc.
        for fmt in (f"{layer_idx:02}", f"{layer_idx:03}", f"{layer_idx:04}", str(layer_idx)):
            for ext in layer_exts:
                yield osp.join(layers_dir, f"{fmt}{ext}")

    if layer_range is not None:
        start_idx, end_idx = layer_range

    idxs = list(range(int(start_idx), int(end_idx)))
    if len(idxs) < CFG.in_chans:
        raise ValueError(
            f"{fragment_id}: expected at least {CFG.in_chans} layers, got {len(idxs)} from range {start_idx}-{end_idx}"
        )
    if len(idxs) > CFG.in_chans:
        start = max(0, (len(idxs) - CFG.in_chans) // 2)
        idxs = idxs[start:start + CFG.in_chans]
    if len(idxs) != CFG.in_chans:
        raise ValueError(
            f"{fragment_id}: expected {CFG.in_chans} layers after cropping, got {len(idxs)} from range {start_idx}-{end_idx}"
        )

    layer_read_workers = int(getattr(CFG, "layer_read_workers", 1) or 1)
    layer_read_workers = max(1, min(layer_read_workers, len(idxs)))

    if "frag" in fragment_id:
        images_list = []
        for i in idxs:
            image = None
            for image_path in _iter_layer_paths(i):
                image = _read_gray(image_path)
                if image is not None:
                    break
            if image is None:
                raise FileNotFoundError(
                    f"Could not read layer for {fragment_id}: {layers_dir}/{i}.[tif|tiff|png|jpg|jpeg]"
                )

            pad0 = (256 - image.shape[0] % 256)
            pad1 = (256 - image.shape[1] % 256)

            image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
            image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_AREA)
            np.clip(image, 0, 200, out=image)
            if fragment_id == '20230827161846':
                image = cv2.flip(image, 0)
            images_list.append(image)

        images = np.stack(images_list, axis=2)
        del images_list
        return images

    first = None
    for image_path in _iter_layer_paths(idxs[0]):
        first = _read_gray(image_path)
        if first is not None:
            break
    if first is None:
        raise FileNotFoundError(
            f"Could not read layer for {fragment_id}: {layers_dir}/{idxs[0]}.[tif|tiff|png|jpg|jpeg]"
        )

    base_h, base_w = first.shape[:2]
    pad0 = (256 - base_h % 256)
    pad1 = (256 - base_w % 256)
    out_h = base_h + pad0
    out_w = base_w + pad1

    images = np.zeros((out_h, out_w, len(idxs)), dtype=first.dtype)
    np.clip(first, 0, 200, out=first)
    if fragment_id == '20230827161846':
        first = cv2.flip(first, 0)
    images[:base_h, :base_w, 0] = first

    def _load_and_write(task):
        chan, i = task
        img = None
        for image_path in _iter_layer_paths(i):
            img = _read_gray(image_path)
            if img is not None:
                break
        if img is None:
            raise FileNotFoundError(
                f"Could not read layer for {fragment_id}: {layers_dir}/{i}.[tif|tiff|png|jpg|jpeg]"
            )
        if img.shape[0] != base_h or img.shape[1] != base_w:
            raise ValueError(
                f"{fragment_id}: layer {i:02} has shape {img.shape} but expected {(base_h, base_w)}"
            )
        np.clip(img, 0, 200, out=img)
        if fragment_id == '20230827161846':
            img = cv2.flip(img, 0)
        images[:base_h, :base_w, chan] = img
        return None

    tasks = [(chan, i) for chan, i in enumerate(idxs[1:], start=1)]
    if tasks:
        if layer_read_workers > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=layer_read_workers) as executor:
                list(executor.map(_load_and_write, tasks))
        else:
            for task in tasks:
                _load_and_write(task)

    return images


def read_image_mask(
    fragment_id,
    start_idx=15,
    end_idx=45,
    *,
    layer_range=None,
    reverse_layers=False,
    label_suffix="",
    mask_suffix="",
    images=None,
):
    if images is None:
        images = read_image_layers(
            fragment_id,
            start_idx=start_idx,
            end_idx=end_idx,
            layer_range=layer_range,
        )

    if reverse_layers or fragment_id in REVERSE_LAYER_FRAGMENT_IDS_FALLBACK:
        images = images[:, :, ::-1]

    label_base = f"train_scrolls/{fragment_id}/{fragment_id}_inklabels{label_suffix}"
    mask = _read_gray(f"{label_base}.png")
    if mask is None:
        mask = _read_gray(f"{label_base}.tiff")
    if mask is None:
        mask = _read_gray(f"{label_base}.tif")
    if mask is None:
        raise FileNotFoundError(f"Could not read label for {fragment_id}: {label_base}.png/.tif/.tiff")

    mask_base = f"train_scrolls/{fragment_id}/{fragment_id}_mask{mask_suffix}"
    fragment_mask = _read_gray(f"{mask_base}.png")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tiff")
    if fragment_mask is None:
        fragment_mask = _read_gray(f"{mask_base}.tif")
    if fragment_mask is None:
        raise FileNotFoundError(f"Could not read mask for {fragment_id}: {mask_base}.png/.tif/.tiff")
    if fragment_id=='20230827161846':
        fragment_mask=cv2.flip(fragment_mask,0)

    def _assert_bottom_right_pad_compatible(a_name, a_hw, b_name, b_hw, multiple):
        a_h, a_w = [int(x) for x in a_hw]
        b_h, b_w = [int(x) for x in b_hw]

        def _check_dim(dim_name, a_dim, b_dim):
            small = min(a_dim, b_dim)
            big = max(a_dim, b_dim)
            padded = ((small + multiple - 1) // multiple) * multiple
            allowed = {small, padded}
            if small % multiple == 0:
                allowed.add(small + multiple)  # supports the legacy "always pad one block" variant

            if big not in allowed:
                raise ValueError(
                    f"{fragment_id}: {a_name} {a_hw} vs {b_name} {b_hw} mismatch. "
                    f"Only bottom/right padding to a multiple of {multiple} is allowed "
                    f"(see inference_resnet3d.py). Got {dim_name}={a_dim} vs {b_dim}."
                )

        _check_dim("height", a_h, b_h)
        _check_dim("width", a_w, b_w)

    if "frag" not in fragment_id:
        pad_multiple = 256
        _assert_bottom_right_pad_compatible("image", images.shape[:2], "label", mask.shape[:2], pad_multiple)
        _assert_bottom_right_pad_compatible("image", images.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)
        _assert_bottom_right_pad_compatible("label", mask.shape[:2], "mask", fragment_mask.shape[:2], pad_multiple)

    if "frag" in fragment_id:
        pad0 = max(0, images.shape[0] * 2 - fragment_mask.shape[0])
        pad1 = max(0, images.shape[1] * 2 - fragment_mask.shape[1])
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    else:
        fragment_mask_padded = np.zeros((images.shape[0], images.shape[1]), dtype=fragment_mask.dtype)
        h = min(fragment_mask.shape[0], fragment_mask_padded.shape[0])
        w = min(fragment_mask.shape[1], fragment_mask_padded.shape[1])
        fragment_mask_padded[:h, :w] = fragment_mask[:h, :w]
        fragment_mask = fragment_mask_padded
        del fragment_mask_padded

    kernel = np.ones((16,16),np.uint8)
    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)

    target_h = min(images.shape[0], mask.shape[0], fragment_mask.shape[0])
    target_w = min(images.shape[1], mask.shape[1], fragment_mask.shape[1])
    if target_h <= 0 or target_w <= 0:
        raise ValueError(
            f"{fragment_id}: empty shapes after alignment (images={images.shape}, label={mask.shape}, mask={fragment_mask.shape})"
        )

    images = images[:target_h, :target_w]
    mask = mask[:target_h, :target_w]
    fragment_mask = fragment_mask[:target_h, :target_w]

    mask = mask.astype('float32')
    mask/=255
    if images.shape[0] != mask.shape[0] or images.shape[1] != mask.shape[1]:
        raise ValueError(f"{fragment_id}: label shape {mask.shape} does not match image shape {images.shape[:2]}")
    return images, mask,fragment_mask

def get_train_valid_dataset(fragment_ids=None, *, segments_metadata=None, valid_id=None):
    train_images = []
    train_masks = []

    valid_images = []
    valid_masks = []
    valid_xyxys = []

    if fragment_ids is None:
        fragment_ids = DEFAULT_FRAGMENT_IDS
    if segments_metadata is None:
        segments_metadata = {}
    if valid_id is None:
        valid_id = CFG.valid_id

    for fragment_id in fragment_ids:  
    # for fragment_id in ['20230820203112','20231005123333']:
#,
        
    # for fragment_id in ['20230522181603','20230702185752','20230827161847','20230909121925','20230905134255','20230904135535']:
        print('reading ',fragment_id)
        seg_meta = segments_metadata.get(fragment_id, {})
        image, mask, fragment_mask = read_image_mask(
            fragment_id,
            layer_range=seg_meta.get("layer_range"),
            reverse_layers=bool(seg_meta.get("reverse_layers", False)),
        )
        stride= CFG.stride
        x1_list = list(range(0, image.shape[1]-CFG.tile_size+1,stride))
        y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, stride))
        windows_dict={}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0,CFG.tile_size,CFG.size):
                    for xi in range(0,CFG.tile_size,CFG.size):
                        y1=a+yi
                        x1=b+xi
                        y2=y1+CFG.size
                        x2=x1+CFG.size
                        if fragment_id!=valid_id:
                            if (y1,y2,x1,x2) not in windows_dict:
                                if not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.01):
                                    if not np.any(fragment_mask[a:a+ CFG.tile_size, b:b + CFG.tile_size]==0):
                                        train_images.append(image[y1:y2, x1:x2])
                                        
                                        train_masks.append(mask[y1:y2, x1:x2, None])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                        windows_dict[(y1,y2,x1,x2)]='1'
                        if fragment_id==valid_id:
                            if (y1,y2,x1,x2) not in windows_dict:
                                if not np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]==0):
                                        valid_images.append(image[y1:y2, x1:x2])
                                        valid_masks.append(mask[y1:y2, x1:x2, None])

                                        valid_xyxys.append([x1, y1, x2, y2])
                                        assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                        windows_dict[(y1,y2,x1,x2)]='1'


    return train_images, train_masks, valid_images, valid_masks, valid_xyxys


def build_group_mappings(fragment_ids, segments_metadata, group_key="base_path"):
    fragment_to_group_name = {}
    for fragment_id in fragment_ids:
        seg_meta = (segments_metadata or {}).get(fragment_id, {}) or {}
        group_name = seg_meta.get(group_key)
        if group_name is None:
            group_name = seg_meta.get("base_path", fragment_id)
        fragment_to_group_name[fragment_id] = str(group_name)

    group_names = sorted(set(fragment_to_group_name.values()))
    group_name_to_idx = {name: i for i, name in enumerate(group_names)}
    fragment_to_group_idx = {fid: group_name_to_idx[g] for fid, g in fragment_to_group_name.items()}
    return group_names, group_name_to_idx, fragment_to_group_idx


def extract_patches(image, mask, fragment_mask, *, include_xyxys, filter_empty_tile):
    images = []
    masks = []
    xyxys = []

    stride = CFG.stride
    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))
    windows_dict = {}

    for a in y1_list:
        for b in x1_list:
            if filter_empty_tile and np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.01):
                continue
            if np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0):
                continue

            for yi in range(0, CFG.tile_size, CFG.size):
                for xi in range(0, CFG.tile_size, CFG.size):
                    y1 = a + yi
                    x1 = b + xi
                    y2 = y1 + CFG.size
                    x2 = x1 + CFG.size
                    if (y1, y2, x1, x2) in windows_dict:
                        continue

                    windows_dict[(y1, y2, x1, x2)] = True
                    images.append(image[y1:y2, x1:x2])
                    masks.append(mask[y1:y2, x1:x2, None])
                    if include_xyxys:
                        xyxys.append([x1, y1, x2, y2])
                    assert image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans)

    return images, masks, xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg, xyxys=None, labels=None, groups=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.groups = groups
        
        self.transform = transform
        self.xyxys=xyxys
        self.rotate=CFG.rotate
    def __len__(self):
        return len(self.images)
    def cubeTranslate(self,y):
        x=np.random.uniform(0,1,4).reshape(2,2)
        x[x<.4]=0
        x[x>.633]=2
        x[(x>.4)&(x<.633)]=1
        mask=cv2.resize(x, (x.shape[1]*64,x.shape[0]*64), interpolation = cv2.INTER_AREA)

        
        x=np.zeros((self.cfg.size,self.cfg.size,self.cfg.in_chans)).astype(np.uint8)
        for i in range(3):
            x=np.where(np.repeat((mask==0).reshape(self.cfg.size,self.cfg.size,1), self.cfg.in_chans, axis=2),y[:,:,i:self.cfg.in_chans+i],x)
        return x
    def fourth_augment(self,image):
        image_tmp = np.zeros_like(image)
        cropping_num = random.randint(24, 30)

        start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
        crop_indices = np.arange(start_idx, start_idx + cropping_num)

        start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)

        tmp = np.arange(start_paste_idx, cropping_num)
        np.random.shuffle(tmp)

        cutout_idx = random.randint(0, 2)
        temporal_random_cutout_idx = tmp[:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if random.random() > 0.4:
            image_tmp[..., temporal_random_cutout_idx] = 0
        image = image_tmp
        return image

    def __getitem__(self, idx):
        group_id = 0
        if self.groups is not None:
            group_id = int(self.groups[idx])
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label, xy, group_id
        else:
            image = self.images[idx]
            label = self.labels[idx]
            #3d rotate
            # image=image.transpose(2,1,0)#(c,w,h)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,h,w)
            # image=self.rotate(image=image)['image']
            # image=image.transpose(0,2,1)#(c,w,h)
            # image=image.transpose(2,1,0)#(h,w,c)

            image=self.fourth_augment(image)
            
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//4,self.cfg.size//4)).squeeze(0)
            return image, label, group_id
class CustomDatasetTest(Dataset):
    def __init__(self, images,xyxys, cfg, transform=None):
        self.images = images
        self.xyxys=xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        xy=self.xyxys[idx]
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)

        return image,xy



# from resnetall import generate_model
def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m, mode='fan_out', nonlinearity='relu')
class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class RegressionPLModel(pl.LightningModule):
    def __init__(
        self,
        size=256,
        enc='',
        with_norm=False,
        objective="erm",
        loss_mode="batch",
        robust_step_size=None,
        group_counts=None,
        group_dro_gamma=0.1,
        group_dro_btl=False,
        group_dro_alpha=None,
        group_dro_normalize_loss=False,
        group_dro_min_var_weight=0.0,
        group_dro_adj=None,
        total_steps=780,
        n_groups=1,
        group_names=None,
        stitch_val_dataloader_idx=None,
        stitch_pred_shape=None,
        stitch_segment_id=None,
        stitch_all_val=False,
        stitch_downsample=1,
        stitch_all_val_shapes=None,
        stitch_all_val_segment_ids=None,
    ):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()

        self.n_groups = int(n_groups)
        if group_names is None:
            group_names = [str(i) for i in range(self.n_groups)]
        self.group_names = list(group_names)

        self.group_dro = None
        if str(self.hparams.objective).lower() == "group_dro":
            if robust_step_size is None:
                raise ValueError("group_dro.robust_step_size is required when training.objective is group_dro")
            if group_counts is None:
                raise ValueError("group_counts is required when training.objective is group_dro")

            self.group_dro = GroupDROComputer(
                n_groups=self.n_groups,
                group_counts=group_counts,
                alpha=group_dro_alpha,
                gamma=group_dro_gamma,
                adj=group_dro_adj,
                min_var_weight=group_dro_min_var_weight,
                step_size=robust_step_size,
                normalize_loss=group_dro_normalize_loss,
                btl=group_dro_btl,
            )

        self._stitch_downsample = max(1, int(stitch_downsample or 1))
        self._stitch_buffers = {}
        self._stitch_segment_ids = {}

        if bool(stitch_all_val):
            if stitch_all_val_shapes is None or stitch_all_val_segment_ids is None:
                raise ValueError("stitch_all_val requires stitch_all_val_shapes and stitch_all_val_segment_ids")
            if len(stitch_all_val_shapes) != len(stitch_all_val_segment_ids):
                raise ValueError(
                    "stitch_all_val_shapes and stitch_all_val_segment_ids must have the same length "
                    f"(got {len(stitch_all_val_shapes)} vs {len(stitch_all_val_segment_ids)})"
                )

            for loader_idx, (segment_id, shape) in enumerate(zip(stitch_all_val_segment_ids, stitch_all_val_shapes)):
                h = int(shape[0])
                w = int(shape[1])
                ds_h = (h + self._stitch_downsample - 1) // self._stitch_downsample
                ds_w = (w + self._stitch_downsample - 1) // self._stitch_downsample
                self._stitch_buffers[int(loader_idx)] = (
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                )
                self._stitch_segment_ids[int(loader_idx)] = str(segment_id)
        else:
            stitch_enabled = (stitch_val_dataloader_idx is not None) and (stitch_pred_shape is not None)
            if stitch_enabled:
                h = int(stitch_pred_shape[0])
                w = int(stitch_pred_shape[1])
                ds_h = (h + self._stitch_downsample - 1) // self._stitch_downsample
                ds_w = (w + self._stitch_downsample - 1) // self._stitch_downsample
                idx = int(stitch_val_dataloader_idx)
                self._stitch_buffers[idx] = (
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                    np.zeros((ds_h, ds_w), dtype=np.float32),
                )
                self._stitch_segment_ids[idx] = str(stitch_segment_id or idx)

        self._stitch_enabled = len(self._stitch_buffers) > 0

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)

        self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=1039)
        init_ckpt_path = getattr(CFG, "init_ckpt_path", None)
        if not init_ckpt_path:
            backbone_pretrained_path = getattr(CFG, "backbone_pretrained_path", "./r3d50_KM_200ep.pth")
            if not osp.exists(backbone_pretrained_path):
                raise FileNotFoundError(
                    f"Missing backbone pretrained weights: {backbone_pretrained_path}. "
                    "Either place r3d50_KM_200ep.pth next to train_resnet3d.py, set CFG.backbone_pretrained_path, "
                    "or pass --init_ckpt_path to fine-tune from a previous run."
                )
            backbone_ckpt = torch.load(backbone_pretrained_path, map_location="cpu")
            state_dict = backbone_ckpt.get("state_dict", backbone_ckpt)
            conv1_weight = state_dict['conv1.weight']
            state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
            self.backbone.load_state_dict(state_dict, strict=False)
        # self.backbone=InceptionI3d(in_channels=1,num_classes=512,non_local=True)
        # self.backbone.load_state_dict(torch.load('./pretraining_i3d_epoch=3.pt'),strict=False)
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,20,256,256))], upscale=1)

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)



            
    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        
        return pred_mask

    def compute_per_sample_loss_and_dice(self, logits, targets):
        targets = targets.float()

        smooth_factor = 0.25
        soft_targets = (1.0 - targets) * smooth_factor + targets * (1.0 - smooth_factor)

        bce = F.binary_cross_entropy_with_logits(logits, soft_targets, reduction="none")
        bce = bce.mean(dim=(1, 2, 3))

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        eps = 1e-7
        dice = (2 * intersection + eps) / (union + eps)

        dice_loss = 1.0 - dice
        per_sample_loss = 0.5 * dice_loss + 0.5 * bce
        return per_sample_loss, dice, bce, dice_loss

    def compute_batch_loss(self, logits, targets):
        return 0.5 * self.loss_func1(logits, targets) + 0.5 * self.loss_func2(logits, targets)
    
    def training_step(self, batch, batch_idx):
        x, y, g = batch
        outputs = self(x)

        objective = str(self.hparams.objective).lower()
        loss_mode = str(self.hparams.loss_mode).lower()
        g = g.long()

        if objective == "erm":
            if loss_mode == "batch":
                dice_loss = self.loss_func1(outputs, y)
                bce_loss = self.loss_func2(outputs, y)
                loss = 0.5 * dice_loss + 0.5 * bce_loss
                self.log("train/dice_loss", dice_loss, on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/bce_loss", bce_loss, on_step=True, on_epoch=True, prog_bar=False)
            elif loss_mode == "per_sample":
                per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(outputs, y)
                loss = per_sample_loss.mean()
                self.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
                self.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)
            else:
                raise ValueError(f"Unknown training.loss_mode: {self.hparams.loss_mode!r}")
        elif objective == "group_dro":
            if loss_mode != "per_sample":
                raise ValueError("GroupDRO requires training.loss_mode=per_sample")
            if self.group_dro is None:
                raise RuntimeError("GroupDRO objective was set but group_dro computer was not initialized")

            per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(outputs, y)
            robust_loss, group_loss, group_count, _weights = self.group_dro.loss(per_sample_loss, g)
            loss = robust_loss
            self.log("train/dice", per_sample_dice.mean(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/dice_loss", per_sample_dice_loss.mean(), on_step=True, on_epoch=True, prog_bar=False)
            self.log("train/bce_loss", per_sample_bce.mean(), on_step=True, on_epoch=True, prog_bar=False)

            if self.global_step % CFG.print_freq == 0:
                present = group_count > 0
                if present.any():
                    worst_group_loss = group_loss[present].max()
                else:
                    worst_group_loss = group_loss.max()
                self.log("train/worst_group_loss", worst_group_loss, on_step=True, on_epoch=False, prog_bar=False)

                for group_idx, group_name in enumerate(self.group_names):
                    safe_group_name = str(group_name).replace("/", "_")
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/loss",
                        group_loss[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/count",
                        group_count[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
                    self.log(
                        f"train/group_{group_idx}_{safe_group_name}/adv_prob",
                        self.group_dro.adv_probs[group_idx],
                        on_step=True,
                        on_epoch=False,
                    )
        else:
            raise ValueError(f"Unknown training.objective: {self.hparams.objective!r}")

        if torch.isnan(loss):
            print("Loss nan encountered")
        self.log("train/total_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss}

    def on_validation_epoch_start(self):
        device = self.device
        self._val_loss_sum = torch.tensor(0.0, device=device)
        self._val_dice_sum = torch.tensor(0.0, device=device)
        self._val_bce_sum = torch.tensor(0.0, device=device)
        self._val_dice_loss_sum = torch.tensor(0.0, device=device)
        self._val_count = torch.tensor(0.0, device=device)

        self._val_group_loss_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_dice_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_bce_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_dice_loss_sum = torch.zeros(self.n_groups, device=device)
        self._val_group_count = torch.zeros(self.n_groups, device=device)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, xyxys, g = batch
        outputs = self(x)
        per_sample_loss, per_sample_dice, per_sample_bce, per_sample_dice_loss = self.compute_per_sample_loss_and_dice(outputs, y)

        self._val_loss_sum += per_sample_loss.sum()
        self._val_dice_sum += per_sample_dice.sum()
        self._val_bce_sum += per_sample_bce.sum()
        self._val_dice_loss_sum += per_sample_dice_loss.sum()
        self._val_count += float(per_sample_loss.numel())

        g = g.long()
        self._val_group_loss_sum.scatter_add_(0, g, per_sample_loss)
        self._val_group_dice_sum.scatter_add_(0, g, per_sample_dice)
        self._val_group_bce_sum.scatter_add_(0, g, per_sample_bce)
        self._val_group_dice_loss_sum.scatter_add_(0, g, per_sample_dice_loss)
        self._val_group_count.scatter_add_(0, g, torch.ones_like(per_sample_loss, dtype=self._val_group_count.dtype))

        if self._stitch_enabled and int(dataloader_idx) in self._stitch_buffers:
            pred_buf, count_buf = self._stitch_buffers[int(dataloader_idx)]
            ds = self._stitch_downsample

            y_preds = torch.sigmoid(outputs).to('cpu')
            for i, (x1, y1, x2, y2) in enumerate(xyxys):
                x1 = int(x1)
                y1 = int(y1)
                x2 = int(x2)
                y2 = int(y2)

                x1_ds = x1 // ds
                y1_ds = y1 // ds
                x2_ds = (x2 + ds - 1) // ds
                y2_ds = (y2 + ds - 1) // ds
                target_h = y2_ds - y1_ds
                target_w = x2_ds - x1_ds
                if target_h <= 0 or target_w <= 0:
                    continue

                pred_patch = y_preds[i].unsqueeze(0).float()
                if pred_patch.shape[-2:] != (target_h, target_w):
                    pred_patch = F.interpolate(
                        pred_patch,
                        size=(target_h, target_w),
                        mode="bilinear",
                        align_corners=False,
                    )

                pred_buf[y1_ds:y2_ds, x1_ds:x2_ds] += pred_patch.squeeze(0).squeeze(0).numpy()
                count_buf[y1_ds:y2_ds, x1_ds:x2_ds] += 1.0

        return {"loss": per_sample_loss.mean()}
    
    def on_validation_epoch_end(self):
        if self._val_count.item() > 0:
            avg_loss = self._val_loss_sum / self._val_count
            avg_dice = self._val_dice_sum / self._val_count
            avg_bce = self._val_bce_sum / self._val_count
            avg_dice_loss = self._val_dice_loss_sum / self._val_count
        else:
            avg_loss = torch.tensor(0.0, device=self.device)
            avg_dice = torch.tensor(0.0, device=self.device)
            avg_bce = torch.tensor(0.0, device=self.device)
            avg_dice_loss = torch.tensor(0.0, device=self.device)

        group_count = self._val_group_count
        group_loss = self._val_group_loss_sum / group_count.clamp_min(1)
        group_dice = self._val_group_dice_sum / group_count.clamp_min(1)
        group_bce = self._val_group_bce_sum / group_count.clamp_min(1)
        group_dice_loss = self._val_group_dice_loss_sum / group_count.clamp_min(1)

        present = group_count > 0
        if present.any():
            worst_group_loss = group_loss[present].max()
            worst_group_dice = group_dice[present].min()
        else:
            worst_group_loss = group_loss.max()
            worst_group_dice = group_dice.min()

        self.log("val/avg_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("val/worst_group_loss", worst_group_loss, on_epoch=True, prog_bar=True)
        self.log("val/avg_dice", avg_dice, on_epoch=True, prog_bar=False)
        self.log("val/worst_group_dice", worst_group_dice, on_epoch=True, prog_bar=False)
        self.log("val/avg_bce_loss", avg_bce, on_epoch=True, prog_bar=False)
        self.log("val/avg_dice_loss", avg_dice_loss, on_epoch=True, prog_bar=False)

        for group_idx, group_name in enumerate(self.group_names):
            safe_group_name = str(group_name).replace("/", "_")
            self.log(f"val/group_{group_idx}_{safe_group_name}/loss", group_loss[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/dice", group_dice[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/bce_loss", group_bce[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/dice_loss", group_dice_loss[group_idx], on_epoch=True)
            self.log(f"val/group_{group_idx}_{safe_group_name}/count", group_count[group_idx], on_epoch=True)

        if self._stitch_enabled and self._stitch_buffers:
            sanity_checking = bool(self.trainer is not None and getattr(self.trainer, "sanity_checking", False))

            images = []
            captions = []
            for loader_idx, (pred_buf, count_buf) in self._stitch_buffers.items():
                stitched = np.divide(
                    pred_buf,
                    count_buf,
                    out=np.zeros_like(pred_buf),
                    where=count_buf != 0,
                )
                images.append(np.clip(stitched, 0, 1))
                segment_id = self._stitch_segment_ids.get(loader_idx, str(loader_idx))
                captions.append(f"{segment_id} (ds={self._stitch_downsample})")

            if (not sanity_checking) and (self.trainer is None or self.trainer.is_global_zero):
                if isinstance(self.logger, WandbLogger):
                    self.logger.log_image(key="masks", images=images, caption=captions)

            # reset stitch buffers
            for pred_buf, count_buf in self._stitch_buffers.values():
                pred_buf.fill(0)
                count_buf.fill(0)
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=CFG.lr,
            pct_start=float(getattr(CFG, "onecycle_pct_start", 0.15)),
            steps_per_epoch=self.hparams.total_steps,
            epochs=CFG.epochs,
            div_factor=float(getattr(CFG, "onecycle_div_factor", 25.0)),
            final_div_factor=float(getattr(CFG, "onecycle_final_div_factor", 1e2)),
        )
        # scheduler = get_scheduler(CFG, optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }



class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    """
    https://www.kaggle.com/code/underwearfitting/single-fold-training-of-resnet200d-lb0-965
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(
            optimizer, multiplier, total_epoch, after_scheduler)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def get_scheduler(cfg, optimizer):
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 50, eta_min=1e-6)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=1.0, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
   



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
    parser.add_argument("--stitch_all_val", action="store_true")
    parser.add_argument("--stitch_downsample", type=int, default=None)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--accumulate_grad_batches", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    log(f"start pid={os.getpid()} cwd={os.getcwd()}")
    log(
        "args "
        f"metadata_json={args.metadata_json!r} valid_id={args.valid_id!r} outputs_path={args.outputs_path!r} "
        f"devices={args.devices} accelerator={args.accelerator!r} precision={args.precision!r} "
        f"run_name={args.run_name!r} init_ckpt_path={args.init_ckpt_path!r} resume_from_ckpt={args.resume_from_ckpt!r}"
    )
    try:
        log(
            f"torch cuda_available={torch.cuda.is_available()} cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
            f"device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
        )
    except Exception:
        pass

    base_config = {}
    if args.metadata_json:
        metadata_path = args.metadata_json
        if not osp.isabs(metadata_path):
            if not osp.exists(metadata_path):
                metadata_path = osp.join(osp.dirname(__file__), metadata_path)
        log(f"loading metadata_json={metadata_path}")
        base_config = load_metadata_json(metadata_path)

    base_config = base_config or {}
    base_config.setdefault("training", {})
    base_config["training"].setdefault("objective", getattr(CFG, "objective", "erm"))
    base_config["training"].setdefault("sampler", getattr(CFG, "sampler", "shuffle"))
    base_config["training"].setdefault("loss_mode", getattr(CFG, "loss_mode", "batch"))
    base_config["training"].setdefault("save_every_epoch", getattr(CFG, "save_every_epoch", False))
    base_config.setdefault("group_dro", {})
    base_config["group_dro"].setdefault("group_key", "base_path")

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
        wandb_logger_kwargs = {k: v for k, v in wandb_logger_kwargs.items() if k in wandb_logger_sig.parameters}
    except Exception:
        wandb_logger_kwargs = {}

    log(
        "wandb init "
        f"project={wandb_project!r} entity={wandb_entity!r} group={wandb_group!r} tags={wandb_tags} "
        f"mode={os.environ.get('WANDB_MODE')!r}"
    )
    wandb_t0 = time.time()
    wandb_logger = WandbLogger(project=wandb_project, name=args.run_name, **wandb_logger_kwargs)
    log(f"wandb ready in {time.time() - wandb_t0:.1f}s")

    try:
        wandb_overrides = unflatten_dict(dict(wandb_logger.experiment.config))
    except Exception:
        wandb_overrides = {}

    merged_config = json.loads(json.dumps(base_config))
    deep_merge_dict(merged_config, wandb_overrides)

    apply_metadata_hyperparameters(CFG, merged_config)
    log(
        "config "
        f"objective={CFG.objective} sampler={CFG.sampler} loss_mode={CFG.loss_mode} "
        f"epochs={CFG.epochs} lr={CFG.lr} batch={CFG.train_batch_size} "
        f"accumulate_grad_batches={CFG.accumulate_grad_batches} "
        f"num_workers={CFG.num_workers} layer_read_workers={getattr(CFG, 'layer_read_workers', 1)} "
        f"stitch_all_val={getattr(CFG, 'stitch_all_val', False)} stitch_downsample={getattr(CFG, 'stitch_downsample', 1)}"
    )

    try:
        wandb_logger.experiment.config.update(merged_config, allow_val_change=True)
    except Exception:
        pass

    segments_metadata = merged_config.get("segments", {}) or {}
    fragment_ids = list(segments_metadata.keys()) or DEFAULT_FRAGMENT_IDS

    training_cfg = merged_config.get("training", {}) or {}
    train_fragment_ids = training_cfg.get("train_segments")
    val_fragment_ids = training_cfg.get("val_segments")
    if train_fragment_ids is None:
        train_fragment_ids = fragment_ids
    if val_fragment_ids is None:
        val_fragment_ids = fragment_ids
    train_fragment_ids = list(train_fragment_ids)
    val_fragment_ids = list(val_fragment_ids)
    log(
        "segments "
        f"train={len(train_fragment_ids)} val={len(val_fragment_ids)} "
        f"valid_id={CFG.valid_id!r}"
    )

    if segments_metadata:
        missing_train = sorted(set(train_fragment_ids) - set(fragment_ids))
        missing_val = sorted(set(val_fragment_ids) - set(fragment_ids))
        if missing_train:
            raise ValueError(f"training.train_segments contains unknown segment ids: {missing_train}")
        if missing_val:
            raise ValueError(f"training.val_segments contains unknown segment ids: {missing_val}")

    group_dro_cfg = merged_config.get("group_dro", {}) or {}
    group_key = group_dro_cfg.get("group_key", "base_path")

    init_ckpt_path = args.init_ckpt_path or training_cfg.get("init_ckpt_path") or training_cfg.get("finetune_from")
    if init_ckpt_path:
        init_ckpt_path = osp.expanduser(str(init_ckpt_path))
        if not osp.isabs(init_ckpt_path):
            init_ckpt_path = osp.join(os.getcwd(), init_ckpt_path)
    CFG.init_ckpt_path = init_ckpt_path

    resume_ckpt_path = args.resume_from_ckpt or training_cfg.get("resume_from_ckpt") or training_cfg.get("resume_ckpt_path")
    if resume_ckpt_path:
        resume_ckpt_path = osp.expanduser(str(resume_ckpt_path))
        if not osp.isabs(resume_ckpt_path):
            resume_ckpt_path = osp.join(os.getcwd(), resume_ckpt_path)
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

    if args.save_every_epoch:
        CFG.save_every_epoch = True
    if args.accumulate_grad_batches is not None:
        CFG.accumulate_grad_batches = int(args.accumulate_grad_batches)

    if args.outputs_path is not None:
        CFG.outputs_path = str(args.outputs_path)
    if args.stitch_all_val:
        CFG.stitch_all_val = True
    if args.stitch_downsample is not None:
        CFG.stitch_downsample = max(1, int(args.stitch_downsample))

    run_slug = args.run_name or f"{CFG.objective}_{CFG.sampler}_{CFG.loss_mode}_stitch={CFG.valid_id}"
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

    try:
        if wandb_group is not None:
            wandb_logger.experiment.group = str(wandb_group)
        if wandb_tags:
            wandb_logger.experiment.tags = wandb_tags
        if args.run_name is None:
            wandb_logger.experiment.name = f"{run_slug}_{run_id}"
    except Exception:
        pass

    CFG.outputs_path = run_dir
    CFG.model_dir = osp.join(run_dir, "checkpoints")
    CFG.figures_dir = osp.join(run_dir, "figures")
    CFG.submission_dir = osp.join(run_dir, "submissions")
    CFG.log_dir = osp.join(run_dir, "logs")

    cfg_init(CFG)
    log(f"dirs checkpoints={CFG.model_dir} logs={CFG.log_dir}")

    torch.set_float32_matmul_precision('medium')

    group_names, _group_name_to_idx, fragment_to_group_idx = build_group_mappings(
        fragment_ids,
        segments_metadata,
        group_key=group_key,
    )

    train_transform = get_transforms(data="train", cfg=CFG)
    valid_transform = get_transforms(data="valid", cfg=CFG)

    train_images = []
    train_masks = []
    train_groups = []

    val_loaders = []
    val_stitch_shapes = []
    val_stitch_segment_ids = []
    stitch_val_dataloader_idx = None
    stitch_pred_shape = None
    stitch_segment_id = None

    log("building datasets")
    train_set = set(train_fragment_ids)
    val_set = set(val_fragment_ids)
    overlap_segments = train_set & val_set
    layers_cache = {}

    for fragment_id in train_fragment_ids:
        seg_meta = segments_metadata.get(fragment_id, {}) or {}
        group_idx = fragment_to_group_idx[fragment_id]
        group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

        t0 = time.time()
        log(f"load train segment={fragment_id} group={group_name}")
        layers = read_image_layers(
            fragment_id,
            layer_range=seg_meta.get("layer_range"),
        )
        if fragment_id in overlap_segments:
            layers_cache[fragment_id] = layers

        image, mask, fragment_mask = read_image_mask(
            fragment_id,
            reverse_layers=bool(seg_meta.get("reverse_layers", False)),
            label_suffix="",
            mask_suffix="",
            images=layers,
        )
        log(
            f"loaded train segment={fragment_id} "
            f"image={tuple(image.shape)} label={tuple(mask.shape)} mask={tuple(fragment_mask.shape)} "
            f"in {time.time() - t0:.1f}s"
        )
        log(f"extract train patches segment={fragment_id}")
        t1 = time.time()
        frag_train_images, frag_train_masks, _ = extract_patches(
            image,
            mask,
            fragment_mask,
            include_xyxys=False,
            filter_empty_tile=True,
        )
        log(f"patches train segment={fragment_id} n={len(frag_train_images)} in {time.time() - t1:.1f}s")
        train_images.extend(frag_train_images)
        train_masks.extend(frag_train_masks)
        train_groups.extend([group_idx] * len(frag_train_images))

    for fragment_id in val_fragment_ids:
        seg_meta = segments_metadata.get(fragment_id, {}) or {}
        group_idx = fragment_to_group_idx[fragment_id]
        group_name = group_names[group_idx] if group_idx < len(group_names) else str(group_idx)

        t0 = time.time()
        log(f"load val segment={fragment_id} group={group_name}")
        layers = layers_cache.get(fragment_id)
        if layers is None:
            layers = read_image_layers(
                fragment_id,
                layer_range=seg_meta.get("layer_range"),
            )
        else:
            log(f"reuse layers cache for val segment={fragment_id}")

        image_val, mask_val, fragment_mask_val = read_image_mask(
            fragment_id,
            reverse_layers=bool(seg_meta.get("reverse_layers", False)),
            label_suffix="_val",
            mask_suffix="_val",
            images=layers,
        )
        log(
            f"loaded val segment={fragment_id} "
            f"image={tuple(image_val.shape)} label={tuple(mask_val.shape)} mask={tuple(fragment_mask_val.shape)} "
            f"in {time.time() - t0:.1f}s"
        )
        log(f"extract val patches segment={fragment_id}")
        t1 = time.time()
        frag_val_images, frag_val_masks, frag_val_xyxys = extract_patches(
            image_val,
            mask_val,
            fragment_mask_val,
            include_xyxys=True,
            filter_empty_tile=False,
        )
        log(f"patches val segment={fragment_id} n={len(frag_val_images)} in {time.time() - t1:.1f}s")
        if len(frag_val_images) == 0:
            continue

        frag_val_xyxys = np.stack(frag_val_xyxys) if len(frag_val_xyxys) > 0 else np.zeros((0, 4), dtype=np.int64)
        frag_val_groups = [group_idx] * len(frag_val_images)
        val_dataset = CustomDataset(
            frag_val_images,
            CFG,
            xyxys=frag_val_xyxys,
            labels=frag_val_masks,
            groups=frag_val_groups,
            transform=valid_transform,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=CFG.valid_batch_size,
            shuffle=False,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        val_loaders.append(val_loader)
        val_stitch_shapes.append(tuple(mask_val.shape))
        val_stitch_segment_ids.append(fragment_id)
        if fragment_id == CFG.valid_id:
            stitch_val_dataloader_idx = len(val_loaders) - 1
            stitch_pred_shape = mask_val.shape
            stitch_segment_id = fragment_id

    log(f"dataset built train_patches={len(train_images)} val_loaders={len(val_loaders)}")
    if len(val_loaders) == 0:
        raise ValueError("No validation data was built (all segments produced 0 validation patches).")

    train_dataset = CustomDataset(
        train_images,
        CFG,
        labels=train_masks,
        groups=train_groups,
        transform=train_transform,
    )

    group_array = torch.as_tensor(train_groups, dtype=torch.long)
    group_counts = torch.bincount(group_array, minlength=len(group_names)).float()
    train_group_counts = [int(x) for x in group_counts.tolist()]
    log(f"train group counts {dict(zip(group_names, train_group_counts))}")

    if CFG.sampler == "shuffle":
        train_sampler = None
        train_shuffle = True
        train_batch_sampler = None
    elif CFG.sampler == "group_balanced":
        group_weights = len(train_dataset) / group_counts.clamp_min(1)
        weights = group_weights[group_array]
        train_sampler = WeightedRandomSampler(weights, len(train_dataset), replacement=True)
        train_shuffle = False
        train_batch_sampler = None
    elif CFG.sampler == "group_stratified":
        train_sampler = None
        train_shuffle = False
        train_batch_sampler = GroupStratifiedBatchSampler(
            train_groups,
            batch_size=CFG.train_batch_size,
            seed=getattr(CFG, "seed", 0),
            drop_last=True,
        )

    if train_batch_sampler is not None:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_batch_sampler,
            num_workers=CFG.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=CFG.train_batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=CFG.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    steps_per_epoch = len(train_loader)
    accum = int(getattr(CFG, "accumulate_grad_batches", 1) or 1)
    if accum > 1:
        steps_per_epoch = int(math.ceil(steps_per_epoch / accum))

    model = RegressionPLModel(
        enc='i3d',
        size=CFG.size,
        objective=CFG.objective,
        loss_mode=CFG.loss_mode,
        robust_step_size=robust_step_size,
        group_counts=train_group_counts,
        group_dro_gamma=group_dro_gamma,
        group_dro_btl=group_dro_btl,
        group_dro_alpha=group_dro_alpha,
        group_dro_normalize_loss=group_dro_normalize_loss,
        group_dro_min_var_weight=group_dro_min_var_weight,
        group_dro_adj=group_dro_adj,
        total_steps=steps_per_epoch,
        n_groups=len(group_names),
        group_names=group_names,
        stitch_val_dataloader_idx=stitch_val_dataloader_idx,
        stitch_pred_shape=stitch_pred_shape,
        stitch_segment_id=stitch_segment_id,
        stitch_all_val=bool(getattr(CFG, "stitch_all_val", False)),
        stitch_downsample=int(getattr(CFG, "stitch_downsample", 1)),
        stitch_all_val_shapes=val_stitch_shapes,
        stitch_all_val_segment_ids=val_stitch_segment_ids,
    )
    if init_ckpt_path:
        log(f"loading init weights from {init_ckpt_path}")
        ckpt = torch.load(init_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict):
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and isinstance(ckpt.get("model_state_dict"), dict):
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        else:
            raise ValueError(f"Unsupported checkpoint format for init_ckpt_path={init_ckpt_path!r}")

        if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        incompat = model.load_state_dict(state_dict, strict=False)
        try:
            missing = len(incompat.missing_keys)
            unexpected = len(incompat.unexpected_keys)
            log(f"loaded init weights (missing_keys={missing}, unexpected_keys={unexpected})")
        except Exception:
            log("loaded init weights")
    try:
        wandb_logger.watch(model, log="all", log_freq=100)
    except Exception:
        pass

    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        logger=wandb_logger,
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
    log("starting trainer.fit")
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=val_loaders, ckpt_path=resume_ckpt_path)
    wandb.finish()


if __name__ == "__main__":
    main()
