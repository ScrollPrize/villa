import os.path as osp
import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 109951162777600
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '109951162777600'
import torch
import torch.nn as nn
import torch.nn.functional as F


import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
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
import gc
import scipy as sp
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW

import datetime
import segmentation_models_pytorch as smp
import numpy as np
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from models.i3dallnl import InceptionI3d
import torch.nn as nn
import torch
from warmup_scheduler import GradualWarmupScheduler
from scipy import ndimage
from models.resnetall import generate_model

from data import *
class CFG:
    # ============== comp exp name =============
    comp_name = 'vesuvius'

    # comp_dir_path = './'
    comp_dir_path = './'
    comp_folder_name = './'
    # comp_dataset_path = f'{comp_dir_path}datasets/{comp_folder_name}/'
    comp_dataset_path = f'./'

    exp_name = 'pretraining_all_3d_decoder_v2'

    # ============== pred target =============
    target_size = 1

    # ============== model cfg =============
    model_name = 'Unet'
    # backbone = 'efficientnet-b0'
    # backbone = 'se_resnext50_32x4d'
    backbone='resnet3d'
    in_chans = 62 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 256
    tile_size = 512
    stride = tile_size // 4

    train_batch_size = 12 # 32
    valid_batch_size = train_batch_size*2
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 2e-5
    # ============== fold =============
    valid_id = '20230820203112'

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
    num_workers = 8

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
    train_aug_list = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.75),
        A.RandomGamma(gamma_limit=(70, 150), p=0.5),
        A.ShiftScaleRotate(rotate_limit=360, shift_limit=0.15, scale_limit=0.1, p=0.75),
        A.OneOf([
            A.GaussNoise(var_limit=[10, 50]),
            A.GaussianBlur(),
            A.MotionBlur(),
        ], p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2),
                        mask_fill_value=0, p=0.5),
        A.Normalize(mean=[0] * in_chans, std=[1] * in_chans),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]
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
cfg_init(CFG)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def read_image_mask(fragment_id,start_idx=15,end_idx=45):

    images = []

    # idxs = range(65)
    mid = 65 // 2
    start = mid - CFG.in_chans // 2
    end = mid + CFG.in_chans // 2
    idxs = range(start_idx, end_idx)

    for i in idxs:

        image = cv2.imread(f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)

        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)

        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
        # image = ndimage.median_filter(image, size=5)

        # image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        if 'frag' in fragment_id:
            image = cv2.resize(image, (image.shape[1]//2,image.shape[0]//2), interpolation = cv2.INTER_AREA)
        image=np.clip(image,0,200)
        if fragment_id=='20230827161846':
            image=cv2.flip(image,0)
        images.append(image)
    images = np.stack(images, axis=2)
    if fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193500P4um','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:

        images=images[:,:,::-1]

    if fragment_id in ['20231022170901','20231022170900']:
        mask = cv2.imread( f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.tiff", 0)
    else:
        mask = cv2.imread( f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)

    # mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    fragment_mask=cv2.imread(f"train_scrolls/{fragment_id}/{fragment_id}_mask.png", 0)
    if fragment_id=='20230827161846':
        fragment_mask=cv2.flip(fragment_mask,0)

    fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

    kernel = np.ones((16,16),np.uint8)
    if 'frag' in fragment_id:
        fragment_mask = cv2.resize(fragment_mask, (fragment_mask.shape[1]//2,fragment_mask.shape[0]//2), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(mask , (mask.shape[1]//2,mask.shape[0]//2), interpolation = cv2.INTER_AREA)

    mask = mask.astype('float32')
    mask/=255
    assert images.shape[0]==mask.shape[0]
    return images, mask,fragment_mask

def get_train_valid_dataset():
    valid_xyxys = []
    # Stream tiles to disk to avoid OOM with many segments
    _cache = '/tmp/vesuvius_tile_cache'
    os.makedirs(_cache, exist_ok=True)

    _tile_shape = (CFG.size, CFG.size, CFG.in_chans)
    _mask_shape = (CFG.size, CFG.size, 1)
    _tile_bytes = CFG.size * CFG.size * CFG.in_chans      # uint8
    _mask_bytes = CFG.size * CFG.size * 1 * 4              # float32

    # Reuse existing cache if files are present and non-empty
    _tr_img_path = f'{_cache}/train_img.bin'
    _va_img_path = f'{_cache}/valid_img.bin'
    _xyxys_path = f'{_cache}/valid_xyxys.npy'
    if (os.path.exists(_tr_img_path) and os.path.getsize(_tr_img_path) > 0 and
        os.path.exists(_va_img_path) and os.path.getsize(_va_img_path) > 0 and
        os.path.exists(_xyxys_path)):
        _n_train = os.path.getsize(_tr_img_path) // _tile_bytes
        _n_valid = os.path.getsize(_va_img_path) // _tile_bytes
        valid_xyxys = np.load(_xyxys_path).tolist()
        print(f'Reusing tile cache: {_n_train} train, {_n_valid} valid tiles', flush=True)
        train_images = np.memmap(_tr_img_path, dtype=np.uint8, mode='r', shape=(_n_train, *_tile_shape))
        train_masks = np.memmap(f'{_cache}/train_mask.bin', dtype=np.float32, mode='r', shape=(_n_train, *_mask_shape))
        valid_images = np.memmap(_va_img_path, dtype=np.uint8, mode='r', shape=(_n_valid, *_tile_shape))
        valid_masks = np.memmap(f'{_cache}/valid_mask.bin', dtype=np.float32, mode='r', shape=(_n_valid, *_mask_shape))
        return train_images, train_masks, valid_images, valid_masks, valid_xyxys

    _tr_img_f = open(_tr_img_path, 'wb')
    _tr_msk_f = open(f'{_cache}/train_mask.bin', 'wb')
    _va_img_f = open(_va_img_path, 'wb')
    _va_msk_f = open(f'{_cache}/valid_mask.bin', 'wb')
    _n_train = 0
    _n_valid = 0
    for fragment_id in ['l5_0_crop','l2_0_crop','w030_202601301912','w029_202601261946','w028_20260115221','500p2a','658','-1','l_2','l_3','l_4','l_6','w00','auto_grown_20260220174252405','auto_grown_20260220144552896','46527_2um_try2','20250910185200','Man5outer','auto_grown_20250918234353791','auto_grown_20250919055754487_inp_hr','auto_grown_20250919061352722_inp_hr','w00_sean_240226_front','auto_grown_20251124-w020','auto_grown_20260226121302716_abf']:
        print('reading ',fragment_id)
        if fragment_id=='frag4':
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(22,46),
                    reverse_layers=False,
                    base_path='train_scrolls',
                    tile_size=CFG.tile_size,
                    xyz_scale=2,
                           )
        elif 'Frag' in fragment_id or fragment_id in ['20241108120732','20241108111522','20241108115232','20241113090990','20241113080880','20241113070770','z_dbg_gen_00668_inp_hr']:
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(24,40),
                    reverse_layers=False,
                    base_path='train_scrolls',
                    tile_size=CFG.tile_size,
                    xyz_scale=3.9007078344930277,
                           )
        elif '08312025_l2_0' in fragment_id:
            segment=Segment(
                        segment_id=fragment_id,
                            layer_range=(8,24),
                            reverse_layers=False,
                            base_path='0139_traces',
                            tile_size=CFG.tile_size,
                            xyz_scale=3.9007078344930277,
                                   )
        elif fragment_id in ['l2_0_crop','l5_0_crop','auto_grown_20251124_w025-sm-sb','auto_grown_20251201_w027-sm','w030_202601301912','w028_20260115221','w025_2025083102_crop','w012_2025010815','w017_2025010815','w025_2025083102_crop','w016_2025010815','w029_202601261946'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='2um_dataset/0139_2um',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.25,
                           )
        elif fragment_id in ['auto_grown_20251218010446110-w0','ortho2'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(32,94),
                    reverse_layers=False,
                    base_path='PherMANBp_2um',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.25,
                           )
        elif fragment_id in ['20240618142020','20240618142022_flat'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='s3_2um',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.25,
                           )
        elif fragment_id in ['auto_grown_20260220174252405','auto_grown_20260220144552896','w00'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='2um_dataset/841_2um',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.25,
                           )
        elif fragment_id in ['46527_2um_try2'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=True,
                    base_path='2um_dataset/814_2um',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.25,
                           )
        elif fragment_id in ['500p2a','658'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=True,
                    base_path='2um_dataset',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.25,
                           )
        elif fragment_id in ['20250910185200'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(33,95),
                    reverse_layers=False,
                    base_path='2um_dataset',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.258,
                           )
        elif fragment_id in ['auto_grown_20250918234353791','auto_grown_20250919055754487_inp_hr','auto_grown_20250919061352722_inp_hr'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='2um_dataset/9b_2um',
                    tile_size=CFG.tile_size,
                           )
        elif fragment_id in ['w00_sean_240226_front'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=True,
                    base_path='2um_dataset/manb_2um',
                    tile_size=CFG.tile_size,
                           )
        elif fragment_id in ['auto_grown_20251124-w020'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=True,
                    base_path='2um_dataset/1451_2um',
                    tile_size=CFG.tile_size,
                           )
        elif fragment_id in ['auto_grown_20260226121302716_abf'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=True,
                    base_path='2um_dataset/814_2um',
                    tile_size=CFG.tile_size,
                           )
        elif fragment_id in ['auto_grown_20250926162450723'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='841',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.258,
                           )
        elif fragment_id in ['l2_0'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='2um_dataset/0139_2um',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.258,
                           )

        elif fragment_id in ['l_2','l_3','l_4','l_6','l_1','l_7'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='2um_dataset/s4_2um',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.258,
                           )
        elif fragment_id in ['-1','-2'] :
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='2um_dataset',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.25,
                           )
        elif fragment_id == 'Man5outer':
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=True,
                    base_path='Man5',
                    tile_size=CFG.tile_size,
                           )
        elif fragment_id in ['2um_44kev_0.22m','2um_43kev_0.22m','2um_62kev_0.22m','2um_77kev_0.35m']:
            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(2,64),
                    reverse_layers=True,
                    base_path='front_multi_energy',
                    tile_size=CFG.tile_size,
                    # xyz_scale=1.46,
                           )
        else:

            segment=Segment(
                segment_id=fragment_id,
                    layer_range=(1,63),
                    reverse_layers=False,
                    base_path='2um_dataset',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.258,
                           )
        image, mask, fragment_mask = segment.get_data()
        print(image.shape, mask.shape, fragment_mask.shape, image.max())

        # Extract valid tiles (deduplicated by position)
        stride = CFG.stride
        x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, stride))
        y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, stride))
        windows_dict = {}
        for a in y1_list:
            for b in x1_list:
                for yi in range(0, CFG.tile_size, CFG.size):
                    for xi in range(0, CFG.tile_size, CFG.size):
                        y1 = a + yi
                        x1 = b + xi
                        y2 = y1 + CFG.size
                        x2 = x1 + CFG.size
                        if fragment_id != CFG.valid_id:
                            if (y1, y2, x1, x2) not in windows_dict:
                                if fragment_id == '500p2a' or not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.01):
                                    if not np.any(fragment_mask[y1:y2, x1:x2] == 0) and image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans):
                                        if mask[y1:y2, x1:x2].shape == (CFG.size, CFG.size):
                                            _tr_img_f.write(image[y1:y2, x1:x2].tobytes())
                                            _tr_msk_f.write(mask[y1:y2, x1:x2, None].tobytes())
                                            _n_train += 1
                                            windows_dict[(y1, y2, x1, x2)] = '1'
                        if fragment_id == CFG.valid_id:
                            if (y1, y2, x1, x2) not in windows_dict:
                                if not np.any(fragment_mask[y1:y2, x1:x2] == 0) and image[y1:y2, x1:x2].shape == (CFG.size, CFG.size, CFG.in_chans) and mask[y1:y2, x1:x2].shape == (CFG.size, CFG.size):
                                    _va_img_f.write(image[y1:y2, x1:x2].tobytes())
                                    _va_msk_f.write(mask[y1:y2, x1:x2, None].tobytes())
                                    valid_xyxys.append([x1, y1, x2, y2])
                                    _n_valid += 1
                                    windows_dict[(y1, y2, x1, x2)] = '1'
        print(f'  {fragment_id}: n_train={_n_train}, n_valid={_n_valid}', flush=True)
        del segment, image, mask, fragment_mask
        gc.collect()

    _tr_img_f.close(); _tr_msk_f.close()
    _va_img_f.close(); _va_msk_f.close()
    np.save(_xyxys_path, np.array(valid_xyxys, dtype=np.int32))

    # Memory-map tile files — workers share pages via OS virtual memory, no CoW duplication
    if _n_train > 0:
        train_images = np.memmap(f'{_cache}/train_img.bin', dtype=np.uint8, mode='r', shape=(_n_train, *_tile_shape))
        train_masks = np.memmap(f'{_cache}/train_mask.bin', dtype=np.float32, mode='r', shape=(_n_train, *_mask_shape))
    else:
        train_images, train_masks = [], []
    if _n_valid > 0:
        valid_images = np.memmap(f'{_cache}/valid_img.bin', dtype=np.uint8, mode='r', shape=(_n_valid, *_tile_shape))
        valid_masks = np.memmap(f'{_cache}/valid_mask.bin', dtype=np.float32, mode='r', shape=(_n_valid, *_mask_shape))
    else:
        valid_images, valid_masks = [], []
    print(f'Total: {_n_train} train, {_n_valid} valid tiles (disk-backed memmap)', flush=True)
    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    """Memory-efficient dataset backed by memmap tile arrays."""
    def __init__(self, images, cfg, xyxys=None, labels=None, transform=None):
        self.images = images   # memmap or ndarray, shape (N, H, W, C)
        self.cfg = cfg
        self.labels = labels   # memmap or ndarray, shape (N, H, W, 1)
        self.transform = transform
        self.xyxys = xyxys

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(self.images[idx])
        label = np.array(self.labels[idx])

        if self.transform:
            data = self.transform(image=image, mask=label)
            image = data['image'].unsqueeze(0)
            label = data['mask'][:, ::4, ::4].contiguous()

        if self.xyxys is not None:
            return image, label, self.xyxys[idx]
        return image, label
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



# ======================== 3D UNet Decoder Components ========================

class ResConvBlock3D(nn.Module):
    """Double 3x3x3 conv with GroupNorm, ReLU, and residual shortcut."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv3d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + residual)


class DepthAttentionCollapse(nn.Module):
    """Learned attention over depth dimension: 1x1x1 conv -> softmax -> weighted sum."""
    def __init__(self, in_ch):
        super().__init__()
        self.attn_conv = nn.Conv3d(in_ch, 1, 1)

    def forward(self, x):
        # x: (B, C, D, H, W)
        attn = self.attn_conv(x)              # (B, 1, D, H, W)
        attn = F.softmax(attn, dim=2)         # softmax over depth
        out = (x * attn).sum(dim=2)           # (B, C, H, W)
        return out


class AuxHead(nn.Module):
    """Auxiliary prediction head: adaptive avg pool depth -> Conv2d 1x1 -> bilinear upsample."""
    def __init__(self, in_ch, target_size=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv = nn.Conv2d(in_ch, 1, 1)
        self.target_size = target_size

    def forward(self, x):
        # x: (B, C, D, H, W)
        x = self.pool(x).squeeze(2)          # (B, C, H, W)
        x = self.conv(x)                     # (B, 1, H, W)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False)
        return x


class Decoder3DUNet(nn.Module):
    """
    3D UNet decoder with channel reduction, skip connections, depth attention collapse.

    Encoder features (from ResNet-152 3D):
        Stage 0: (B, 256,  62, 64, 64)
        Stage 1: (B, 512,  31, 32, 32)
        Stage 2: (B, 1024, 16, 16, 16)
        Stage 3: (B, 2048,  8,  8,  8)

    After channel reduction (4x):
        Stage 0: (B, 64,  62, 64, 64)
        Stage 1: (B, 128, 31, 32, 32)
        Stage 2: (B, 256, 16, 16, 16)
        Stage 3: (B, 512,  8,  8,  8)

    Decoder path:
        512@8,8,8 -> up+cat(256) -> 256@16,16,16
        256@16,16,16 -> up+cat(128) -> 128@31,32,32
        128@31,32,32 -> up+cat(64) -> 64@62,64,64
        -> DepthAttentionCollapse -> (B,64,64,64)
        -> Conv2d 1x1 -> (B,1,64,64)
    """
    def __init__(self, encoder_dims=(256, 512, 1024, 2048),
                 decoder_dims=(64, 128, 256, 512), deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Channel reduction: 1x1x1 conv to reduce encoder channels by 4x
        self.channel_reduce = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(enc_d, dec_d, 1, bias=False),
                nn.GroupNorm(min(32, dec_d), dec_d),
                nn.ReLU(inplace=True)
            ) for enc_d, dec_d in zip(encoder_dims, decoder_dims)
        ])

        # Decoder blocks (bottom-up):
        #   Stage 3->2: cat(512, 256)=768  -> 256
        #   Stage 2->1: cat(256, 128)=384  -> 128
        #   Stage 1->0: cat(128, 64) =192  -> 64
        self.decoder_blocks = nn.ModuleList([
            ResConvBlock3D(decoder_dims[i] + decoder_dims[i - 1], decoder_dims[i - 1])
            for i in range(len(decoder_dims) - 1, 0, -1)
        ])

        # Depth attention collapse: (B,64,D,H,W) -> (B,64,H,W)
        self.depth_collapse = DepthAttentionCollapse(decoder_dims[0])

        # Final 2D prediction head
        self.logit = nn.Conv2d(decoder_dims[0], 1, 1)

        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_head_s2 = AuxHead(decoder_dims[2])   # after stage 3->2: 256 ch
            self.aux_head_s1 = AuxHead(decoder_dims[1])   # after stage 2->1: 128 ch

    def forward(self, feat_maps):
        # feat_maps: [stage0, stage1, stage2, stage3] from encoder
        feats = [self.channel_reduce[i](feat_maps[i]) for i in range(4)]

        aux_outputs = []

        # --- Stage 3 -> 2 ---
        x = feats[3]  # (B, 512, 8, 8, 8)
        x = F.interpolate(x, size=feats[2].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, feats[2]], dim=1)
        x = self.decoder_blocks[0](x)  # (B, 256, 16, 16, 16)
        if self.deep_supervision and self.training:
            aux_outputs.append(self.aux_head_s2(x))

        # --- Stage 2 -> 1 ---
        x = F.interpolate(x, size=feats[1].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, feats[1]], dim=1)
        x = self.decoder_blocks[1](x)  # (B, 128, 31, 32, 32)
        if self.deep_supervision and self.training:
            aux_outputs.append(self.aux_head_s1(x))

        # --- Stage 1 -> 0 ---
        x = F.interpolate(x, size=feats[0].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, feats[0]], dim=1)
        x = self.decoder_blocks[2](x)  # (B, 64, 62, 64, 64)

        # Depth attention collapse -> 2D
        x = self.depth_collapse(x)  # (B, 64, 64, 64)

        # Final prediction
        x = self.logit(x)  # (B, 1, 64, 64)

        if self.deep_supervision and self.training:
            return x, aux_outputs
        return x


# ======================== Lightning Model ========================

class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,enc='',with_norm=False,total_steps=780):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)

        self.backbone = generate_model(model_depth=152, n_input_channels=1,forward_features=True,n_classes=1039)
        state_dict=torch.load('./r3d152_KM_200ep.pth')["state_dict"]
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.backbone.load_state_dict(state_dict,strict=False)

        # 3D UNet decoder (replaces 2D Decoder with depth-preserving 3D path)
        self.decoder = Decoder3DUNet(
            encoder_dims=(256, 512, 1024, 2048),
            decoder_dims=(64, 128, 256, 512),
            deep_supervision=True
        )

        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)




    def forward(self, x):
        if x.ndim==4:
            x=x[:,None]
        if self.hparams.with_norm:
            x=self.normalization(x)
        feat_maps = self.backbone(x)
        # Pass full 3D feature maps directly (no depth pooling)
        pred_mask = self.decoder(feat_maps)
        return pred_mask

    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
        if isinstance(outputs, tuple):
            pred, aux_preds = outputs
            main_loss = self.loss_func(pred, y)
            # Auxiliary loss weight decays linearly over first 15 epochs
            aux_weight = max(0.0, 1.0 - self.current_epoch / 15) * 0.4
            aux_loss = sum(self.loss_func(a, y) for a in aux_preds) / len(aux_preds)
            loss1 = main_loss + aux_weight * aux_loss
        else:
            loss1 = self.loss_func(outputs, y)
        if torch.isnan(loss1):
            print("Loss nan encountered")
        self.log("train/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((self.hparams.size, self.hparams.size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}

    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

    def configure_optimizers(self):
        # Two param groups: encoder (low lr) and decoder (high lr)
        encoder_params = list(self.backbone.parameters())
        if self.hparams.with_norm:
            encoder_params += list(self.normalization.parameters())
        decoder_params = list(self.decoder.parameters())

        optimizer = AdamW([
            {'params': encoder_params, 'lr': CFG.lr},
            {'params': decoder_params, 'lr': CFG.lr},
        ], weight_decay=CFG.weight_decay)

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[1e-5, 3e-4],
            pct_start=0.15,
            steps_per_epoch=self.hparams.total_steps,
            epochs=20,
            final_div_factor=1e2
        )
        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]



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




fragment_id = CFG.valid_id

torch.set_float32_matmul_precision('medium')

fragments=['l5_0_crop']
enc_i,enc,fold=0,'i3d',0
for fid in fragments:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'training__valid={fragment_id}_3d_decoder_v2'

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"2um_dataset/0139_2um/{fragment_id}/{fragment_id}_inklabels.png", 0)
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)

    valid_mask_gt = np.pad(valid_mask_gt, [(0, pad0), (0, pad1)], constant_values=0)
    pred_shape=valid_mask_gt.shape
    print(pred_shape)
    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
    print(len(train_images))
    valid_xyxys = np.stack(valid_xyxys) if valid_xyxys else np.empty((0, 4), dtype=np.int64)
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDataset(
        valid_images, CFG, xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

    train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=True)
    valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                persistent_workers=True)

    wandb_logger = WandbLogger(project="ink-experiments",entity='vesuvius-challenge',name=run_slug)
    norm=fold==1
    model=RegressionPLModel(enc='r152',pred_shape=pred_shape,size=CFG.size,total_steps=len(train_loader),with_norm=True)
    print('FOLD : ',fold)
    wandb_logger.watch(model, log="all", log_freq=100)
    multiplicative = lambda epoch: 0.9

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=1,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='auto',
        callbacks=[ModelCheckpoint(filename=f'r152_3ddec_v2_{fid}_{fold}_fr_{enc}'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),

                    ],

    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()
