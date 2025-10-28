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
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
from data import *
from pytorch_metric_learning.losses import NTXentLoss
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
    in_chans = 64 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 256
    tile_size = 512
    stride = tile_size // 6

    train_batch_size = 4 # 32
    valid_batch_size = train_batch_size
    use_amp = True

    scheduler = 'GradualWarmupSchedulerV2'
    # scheduler = 'CosineAnnealingLR'
    epochs = 30 # 30

    # adamW warmupあり
    warmup_factor = 10
    # lr = 1e-4 / warmup_factor
    # lr = 1e-4 / warmup_factor
    lr = 1e-5
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
    num_workers = 16

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
        # A.RandomResizedCrop(
        #     size, size, scale=(0.85, 1.0)),
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.6),

        # A.RandomBrightnessContrast(p=0.75),
        # A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0] * in_chans,
            std= [1] * in_chans
        ),
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

def get_train_valid_dataset(segment_pairs):
    train_images_hr = []
    train_images_lr = []
    train_masks = []
    train_coords_hr = []

    valid_images_hr = []
    valid_images_lr = []
    valid_masks = []
    valid_xyxys = []
    
    for segment_config in segment_pairs:
        fragment_id = segment_config['fragment_id']
        hr_config = segment_config['high_res']
        lr_config = segment_config['low_res']
        
        print('reading ', fragment_id)
        
        segment_hr = Segment(
            segment_id=hr_config['segment_id'],
            layer_range=hr_config['layer_range'],
            reverse_layers=hr_config.get('reverse_layers', False),
            base_path=hr_config['base_path'],
            tile_size=CFG.tile_size,
            xyz_scale=hr_config.get('xyz_scale', 1.0)
        )
        
        segment_lr = Segment(
            segment_id=lr_config['segment_id'],
            layer_range=lr_config['layer_range'],
            reverse_layers=lr_config.get('reverse_layers', False),
            base_path=lr_config['base_path'],
            tile_size=int(CFG.tile_size * segment_config['res_ratio']),
            xyz_scale=lr_config.get('xyz_scale', 1.0)
        )
        
        image_hr, mask, fragment_mask_hr = segment_hr.get_data()
        image_lr, _, fragment_mask_lr = segment_lr.get_data()
        
        print(f"HR shape: {image_hr.shape}, LR shape: {image_lr.shape}, Mask shape: {mask.shape}")
        
        stride = CFG.stride
        x1_list = list(range(0, image_hr.shape[1]-CFG.tile_size+1, stride))
        y1_list = list(range(0, image_hr.shape[0]-CFG.tile_size+1, stride))
        windows_dict = {}
        
        for a in y1_list:
            for b in x1_list:
                for yi in range(0, CFG.tile_size, CFG.size):
                    for xi in range(0, CFG.tile_size, CFG.size):
                        y1_hr = a + yi
                        x1_hr = b + xi
                        y2_hr = y1_hr + CFG.size
                        x2_hr = x1_hr + CFG.size
                        
                        lr_size = 64
                        center_y_hr = (y1_hr + y2_hr) / 2.0
                        center_x_hr = (x1_hr + x2_hr) / 2.0
                        
                        center_y_lr = center_y_hr * segment_config['res_ratio']
                        center_x_lr = center_x_hr * segment_config['res_ratio']
                        
                        y1_lr = int(center_y_lr - lr_size / 2)
                        x1_lr = int(center_x_lr - lr_size / 2)
                        y2_lr = y1_lr + lr_size
                        x2_lr = x1_lr + lr_size
                        
                        if fragment_id != CFG.valid_id:
                            if (y1_hr, y2_hr, x1_hr, x2_hr) not in windows_dict:
                                if not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size] < 0.01):
                                    if not np.any(fragment_mask_hr[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0):
                                        crop_hr = image_hr[y1_hr:y2_hr, x1_hr:x2_hr]
                                        crop_lr = image_lr[y1_lr:y2_lr, x1_lr:x2_lr]
                                        crop_mask_hr = mask[y1_hr:y2_hr, x1_hr:x2_hr, None]
                                        
                                        train_images_hr.append(crop_hr)
                                        train_images_lr.append(crop_lr)
                                        train_masks.append(crop_mask_hr)
                                        windows_dict[(y1_hr, y2_hr, x1_hr, x2_hr)] = '1'
                                        
                        if fragment_id == CFG.valid_id:
                            if not np.any(fragment_mask_hr[a:a + CFG.tile_size, b:b + CFG.tile_size] == 0):
                                crop_hr = image_hr[y1_hr:y2_hr, x1_hr:x2_hr]
                                crop_lr = image_lr[y1_lr:y2_lr, x1_lr:x2_lr]
                                crop_mask_hr = mask[y1_hr:y2_hr, x1_hr:x2_hr, None]
                                
                                valid_images_hr.append(crop_hr)
                                valid_images_lr.append(crop_lr)
                                valid_masks.append(crop_mask_hr)
                                valid_xyxys.append([x1_hr, y1_hr, x2_hr, y2_hr])

    return train_images_hr, train_images_lr, train_masks, valid_images_hr, valid_images_lr, valid_masks, valid_xyxys

def get_transforms(data, cfg, in_chans=None, target_size=None):
    if in_chans is None:
        in_chans = cfg.in_chans
    if target_size is None:
        target_size = cfg.size
    
    if data == 'train':
        train_list = [
            A.Resize(target_size, target_size),
            A.Normalize(
                mean= [0] * in_chans,
                std= [1] * in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ]
        aug = A.Compose(train_list)
    elif data == 'valid':
        valid_list = [
            A.Resize(target_size, target_size),
            A.Normalize(
                mean= [0] * in_chans,
                std= [1] * in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ]
        aug = A.Compose(valid_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images_hr=None, images_lr=None, cfg=None, xyxys=None, labels=None, transform_hr=None, transform_lr=None, dual_res=False, res_ratio=0.2564):
        self.images_hr = images_hr
        self.images_lr = images_lr
        self.cfg = cfg
        self.labels = labels
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.xyxys = xyxys
        self.rotate = CFG.rotate
        self.dual_res = dual_res
        self.res_ratio = res_ratio
        
    def __len__(self):
        return len(self.labels)
        
    def cubeTranslate(self, y):
        x = np.random.uniform(0, 1, 4).reshape(2, 2)
        x[x < .4] = 0
        x[x > .633] = 2
        x[(x > .4) & (x < .633)] = 1
        mask = cv2.resize(x, (x.shape[1]*64, x.shape[0]*64), interpolation=cv2.INTER_AREA)

        x = np.zeros((self.cfg.size, self.cfg.size, self.cfg.in_chans)).astype(np.uint8)
        for i in range(3):
            x = np.where(np.repeat((mask==0).reshape(self.cfg.size, self.cfg.size, 1), self.cfg.in_chans, axis=2), y[:,:,i:self.cfg.in_chans+i], x)
        return x
        
    def fourth_augment(self, image, aug_params=None):
        image_tmp = np.zeros_like(image)
        
        if aug_params is None:
            cropping_num = random.randint(52, 64)
            start_idx = random.randint(0, self.cfg.in_chans - cropping_num)
            start_paste_idx = random.randint(0, self.cfg.in_chans - cropping_num)
            tmp = np.arange(start_paste_idx, cropping_num)
            np.random.shuffle(tmp)
            cutout_idx = random.randint(0, 2)
            do_cutout = random.random() > 0.4
            aug_params = {
                'cropping_num': cropping_num,
                'start_idx': start_idx,
                'start_paste_idx': start_paste_idx,
                'tmp': tmp,
                'cutout_idx': cutout_idx,
                'do_cutout': do_cutout
            }
        
        cropping_num = aug_params['cropping_num']
        start_idx = aug_params['start_idx']
        start_paste_idx = aug_params['start_paste_idx']
        cutout_idx = aug_params['cutout_idx']
        
        crop_indices = np.arange(start_idx, start_idx + cropping_num)
        temporal_random_cutout_idx = aug_params['tmp'][:cutout_idx]

        image_tmp[..., start_paste_idx : start_paste_idx + cropping_num] = image[..., crop_indices]

        if aug_params['do_cutout']:
            image_tmp[..., temporal_random_cutout_idx] = 0
        
        return image_tmp, aug_params

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.xyxys is not None:
            image_lr = self.images_lr[idx]
            xy = self.xyxys[idx]
            mask_lr = cv2.resize(label.squeeze(-1), (image_lr.shape[1], image_lr.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_lr = mask_lr[:, :, None]
            
            if self.transform_lr:
                data = self.transform_lr(image=image_lr, mask=mask_lr)
                image_lr = data['image'].unsqueeze(0)
                label = data['mask']
                label = F.interpolate(label.unsqueeze(0), (64//4, 64//4)).squeeze(0)
            return image_lr, label, xy
        
        else:
            image_hr = self.images_hr[idx]
            image_lr = self.images_lr[idx]
            
            mask_lr = cv2.resize(label.squeeze(-1), (image_lr.shape[1], image_lr.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask_lr = mask_lr[:, :, None]
            
            if self.transform_hr and self.transform_lr:
                data_hr = self.transform_hr(image=image_hr, mask=label)
                data_lr = self.transform_lr(image=image_lr, mask=mask_lr)
                
                image_hr = data_hr['image'].unsqueeze(0)
                image_lr = data_lr['image'].unsqueeze(0)
                label = data_hr['mask']
                label = F.interpolate(label.unsqueeze(0), (64//4, 64//4)).squeeze(0)
            
            return image_lr, image_hr, label
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
# y_preds = F.adaptive_max_pool3d(feat_maps[-1],(1,1,1)).view(b,-1).detach()

class TeacherModel(nn.Module):
    def __init__(self,ckpt_path=''):
        super(TeacherModel,self).__init__()
        self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=1039)
        if ckpt_path:
            checkpoint=torch.load(ckpt_path,weights_only=False)
            self.backbone.load_state_dict(checkpoint['state_dict'],strict=False)
        
        for param in self.parameters():
            param.requires_grad = False
            
    def forward(self,x):
        if x.ndim==4:
            x=x[:,None]
        feat_maps = self.backbone(x)
        feat_maps_pooled = [torch.max(f, dim=2)[0] for f in feat_maps]
        return feat_maps_pooled

class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,enc='',with_norm=False,total_steps=780,teacher_ckpt='',distill_alpha=0.5,student_in_chans=16,teacher_in_chans=64,distill_loss_type='cosine',infonce_temperature=0.1):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
        self.first_batch_logged = False

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        
        self.distill_loss_type = distill_loss_type
        if self.distill_loss_type == 'infonce':
            self.infonce_loss = NTXentLoss(temperature=infonce_temperature)
        
        self.backbone = generate_model(model_depth=50, n_input_channels=1,forward_features=True,n_classes=1039)
        
        if teacher_ckpt:
            self.teacher=TeacherModel(teacher_ckpt)
            self.teacher.eval()
        else:
            self.teacher = None
            
        state_dict=torch.load('./r3d50_KM_200ep.pth')["state_dict"]
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        self.backbone.load_state_dict(state_dict,strict=False)
        
        self.decoder = Decoder(encoder_dims=[x.size(1) for x in self.backbone(torch.rand(1,1,student_in_chans,size,size))], upscale=1)
        self.distill_alpha=distill_alpha
        
        if self.hparams.with_norm:
            self.normalization=nn.BatchNorm3d(num_features=1)

            
    def forward(self, x_student, x_teacher=None):
        if x_student.ndim==4:
            x_student=x_student[:,None]
        if self.hparams.with_norm:
            x_student=self.normalization(x_student)
        
        student_feat_maps = self.backbone(x_student)
        student_feat_maps_pooled = [torch.max(f, dim=2)[0] for f in student_feat_maps]
        pred_mask = self.decoder(student_feat_maps_pooled)
        
        if x_teacher is not None and self.teacher is not None:
            if x_teacher.ndim==4:
                x_teacher=x_teacher[:,None]
            with torch.no_grad():
                teacher_feat_maps_pooled = self.teacher(x_teacher)
            return pred_mask, student_feat_maps_pooled, teacher_feat_maps_pooled
        
        return pred_mask
    
    def training_step(self, batch, batch_idx):
        x_lr, x_hr, y = batch
        
        if not self.first_batch_logged:
            import os
            debug_dir = './debug_images'
            os.makedirs(debug_dir, exist_ok=True)
            
            print("\n" + "="*80)
            print("FIRST BATCH DEBUG - Saving images to ./debug_images/")
            print("="*80)
            
            lr_numpy = x_lr[0, 0].cpu().numpy()
            hr_numpy = x_hr[0, 0].cpu().numpy()
            label_numpy = y[0, 0].cpu().numpy()
            
            num_lr_channels = lr_numpy.shape[0]
            num_hr_channels = hr_numpy.shape[0]
            
            first_lr = int(0)
            mid_lr = int(num_lr_channels // 2)
            last_lr = int(num_lr_channels - 1)
            
            first_hr = int(0)
            mid_hr = int(num_hr_channels // 2)
            last_hr = int(num_hr_channels - 1)
            
            def normalize_for_save(img):
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return (img * 255).astype(np.uint8)
            
            cv2.imwrite(f'{debug_dir}/lr_first_layer_{first_lr}.png', normalize_for_save(lr_numpy[first_lr]))
            cv2.imwrite(f'{debug_dir}/lr_middle_layer_{mid_lr}.png', normalize_for_save(lr_numpy[mid_lr]))
            cv2.imwrite(f'{debug_dir}/lr_last_layer_{last_lr}.png', normalize_for_save(lr_numpy[last_lr]))
            
            cv2.imwrite(f'{debug_dir}/hr_first_layer_{first_hr}.png', normalize_for_save(hr_numpy[first_hr]))
            cv2.imwrite(f'{debug_dir}/hr_middle_layer_{mid_hr}.png', normalize_for_save(hr_numpy[mid_hr]))
            cv2.imwrite(f'{debug_dir}/hr_last_layer_{last_hr}.png', normalize_for_save(hr_numpy[last_hr]))
            
            cv2.imwrite(f'{debug_dir}/label.png', normalize_for_save(label_numpy))
            
            print(f"LR shape: {x_lr.shape} - Saved layers {first_lr}, {mid_lr}, {last_lr}")
            print(f"HR shape: {x_hr.shape} - Saved layers {first_hr}, {mid_hr}, {last_hr}")
            print(f"Label shape: {y.shape}")
            print(f"Images saved to: {debug_dir}/")
            print("="*80 + "\n")
            self.first_batch_logged = True
        
        outputs, student_feats, teacher_feats = self(x_lr, x_hr)
        
        seg_loss = self.loss_func(outputs, y)
        
        if self.distill_loss_type == 'cosine':
            distill_loss = 0.0
            for student_feat, teacher_feat in zip(student_feats, teacher_feats):
                student_feat_upsampled = F.interpolate(student_feat, size=teacher_feat.shape[-2:], mode='bilinear', align_corners=False)
                
                b, c, h, w = student_feat_upsampled.shape
                student_flat = student_feat_upsampled.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c)
                teacher_flat = teacher_feat.reshape(b, c, -1).permute(0, 2, 1).reshape(-1, c)
                
                target = torch.ones(b * h * w, device=student_feat_upsampled.device)
                distill_loss += F.cosine_embedding_loss(student_flat, teacher_flat, target)
            
            distill_loss = distill_loss / len(student_feats)
            
        elif self.distill_loss_type == 'infonce':
            distill_loss = 0.0
            for student_feat, teacher_feat in zip(student_feats, teacher_feats):
                student_feat_upsampled = F.interpolate(student_feat, size=teacher_feat.shape[-2:], mode='bilinear', align_corners=False)
                
                b, c, h, w = student_feat_upsampled.shape
                
                student_embed = F.adaptive_avg_pool2d(student_feat_upsampled, 1).view(b, c)
                teacher_embed = F.adaptive_avg_pool2d(teacher_feat, 1).view(b, c)
                
                embeddings = torch.cat((student_embed, teacher_embed), dim=0)
                
                labels = torch.arange(b, device=embeddings.device)
                labels = torch.cat((labels, labels), dim=0)
                
                distill_loss += self.infonce_loss(embeddings, labels)
            
            distill_loss = distill_loss / len(student_feats)
        else:
            raise ValueError(f"Unknown distill_loss_type: {self.distill_loss_type}")
        
        # total_loss = self.distill_alpha * seg_loss + (1 - self.distill_alpha) * distill_loss
        total_loss = seg_loss

        if torch.isnan(total_loss):
            print("Loss nan encountered")
            
        self.log("train/seg_loss", seg_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"train/distill_loss_{self.distill_loss_type}", distill_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": total_loss}

    def validation_step(self, batch, batch_idx):
        x,y,xyxys= batch
        batch_size = x.size(0)
        
        if batch_idx == 0:
            import os
            debug_dir = './debug_images'
            os.makedirs(debug_dir, exist_ok=True)
            
            print("\n" + "="*80)
            print("VALIDATION BATCH DEBUG - Saving images")
            print("="*80)
            
            val_numpy = x[0, 0].cpu().numpy()
            val_label_numpy = y[0, 0].cpu().numpy()
            
            num_val_channels = val_numpy.shape[0]
            first_val = int(0)
            mid_val = int(num_val_channels // 2)
            last_val = int(num_val_channels - 1)
            
            def normalize_for_save(img):
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                return (img * 255).astype(np.uint8)
            
            cv2.imwrite(f'{debug_dir}/val_first_layer_{first_val}.png', normalize_for_save(val_numpy[first_val]))
            cv2.imwrite(f'{debug_dir}/val_middle_layer_{mid_val}.png', normalize_for_save(val_numpy[mid_val]))
            cv2.imwrite(f'{debug_dir}/val_last_layer_{last_val}.png', normalize_for_save(val_numpy[last_val]))
            cv2.imwrite(f'{debug_dir}/val_label.png', normalize_for_save(val_label_numpy))
            
            print(f"Val Input shape: {x.shape} - Saved layers {first_val}, {mid_val}, {last_val}")
            print(f"Val Label shape: {y.shape}")
            print(f"Val xyxys sample: {xyxys[0]}")
            print(f"Pred shape canvas: {self.mask_pred.shape}")
            print(f"Images saved to: {debug_dir}/")
            print("="*80 + "\n")
        
        outputs = self(x)
        loss1 = self.loss_func(outputs, y)
        y_preds = torch.sigmoid(outputs).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xyxys):
            hr_size = x2 - x1
            upscale_factor = hr_size // self.hparams.size
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),size=(hr_size, hr_size),mode='bilinear').squeeze(0).squeeze(0).numpy()
            self.mask_count[y1:y2, x1:x2] += np.ones((hr_size, hr_size))

        self.log("val/total_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=CFG.lr)
        scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4,pct_start=0.15, steps_per_epoch=self.hparams.total_steps, epochs=50,final_div_factor=1e2)
        # scheduler = get_scheduler(CFG, optimizer)
        return [optimizer],[scheduler]



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

# valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id}_inklabels.png", 0)
# valid_mask_gt=cv2.resize(valid_mask_gt,(valid_mask_gt.shape[1]//2,valid_mask_gt.shape[0]//2),cv2.INTER_AREA)
# pred_shape=valid_mask_gt.shape
torch.set_float32_matmul_precision('medium')

teacher_checkpoint = CFG.model_dir+'0139_wild2_2um_20250807020208_0_fr_i3depoch=3.ckpt'
distill_alpha = 0.5
res_ratio = 0.25636372741307073

distill_loss_type = 'infonce'
infonce_temperature = 0.1

segment_pairs = [
    {
        'fragment_id': 'l2_0_crop',
        'res_ratio': res_ratio,
        'high_res': {
            'segment_id': 'l2_0_crop',
            'layer_range': (0, 64),
            'reverse_layers': False,
            'base_path': '0139_columns',
        },
        'low_res': {
            'segment_id': '08312025_l2_0_crop',
            'layer_range': (8, 24),
            'reverse_layers': False,
            'base_path': '0139_traces',
        }
    },
    {
        'fragment_id': '08312025_l5_0_crop',
        'res_ratio': res_ratio,
        'high_res': {
            'segment_id': 'l5_0_crop',
            'layer_range': (0, 64),
            'reverse_layers': False,
            'base_path': '0139_columns',
        },
        'low_res': {
            'segment_id': '08312025_l5_0_crop',
            'layer_range': (8, 24),
            'reverse_layers': False,
            'base_path': '0139_traces',
        }
    }
]

fragments=['08312025_l5_0_crop']
enc_i,enc,fold=0,'i3d',0

for fid in fragments:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'training__valid={fragment_id}_distillation_alpha{distill_alpha}'

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"0139_traces/{fragment_id}/{fragment_id}_inklabels.png", 0)
    pad0 = (CFG.tile_size - valid_mask_gt.shape[0] % CFG.tile_size)
    pad1 = (CFG.tile_size - valid_mask_gt.shape[1] % CFG.tile_size)

    pred_shape=valid_mask_gt.shape
    print(pred_shape)
    
    train_images_hr, train_images_lr, train_masks, valid_images_hr, valid_images_lr, valid_masks, valid_xyxys = get_train_valid_dataset(segment_pairs)
    print(f"Training samples: {len(train_images_hr)}")
    print(f"HR shape: {train_images_hr[0].shape if len(train_images_hr) > 0 else 'empty'}")
    print(f"LR shape: {train_images_lr[0].shape if len(train_images_lr) > 0 else 'empty'}")
    
    valid_xyxys = np.stack(valid_xyxys)
    
    hr_in_chans = train_images_hr[0].shape[2] if len(train_images_hr) > 0 else 64
    lr_in_chans = train_images_lr[0].shape[2] if len(train_images_lr) > 0 else 16
    lr_size = 64
    
    print(f"HR channels: {hr_in_chans}, LR channels: {lr_in_chans}")
    print(f"HR size: {CFG.size}, LR size: {lr_size}")
    
    train_dataset = CustomDataset(
        images_hr=train_images_hr,
        images_lr=train_images_lr,
        cfg=CFG,
        labels=train_masks,
        transform_hr=get_transforms(data='train', cfg=CFG, in_chans=hr_in_chans, target_size=CFG.size),
        transform_lr=get_transforms(data='train', cfg=CFG, in_chans=lr_in_chans, target_size=lr_size),
        dual_res=True,
        res_ratio=res_ratio
    )
    
    valid_dataset = CustomDataset(
        images_hr=valid_images_hr,
        images_lr=valid_images_lr,
        cfg=CFG,
        xyxys=valid_xyxys,
        labels=valid_masks,
        transform_hr=get_transforms(data='valid', cfg=CFG, in_chans=hr_in_chans, target_size=CFG.size),
        transform_lr=get_transforms(data='valid', cfg=CFG, in_chans=lr_in_chans, target_size=lr_size),
        dual_res=False,
        res_ratio=res_ratio
    )

    train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                )
    valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    wandb_logger = WandbLogger(project="ink-experiments",entity='vesuvius-challenge',name=run_slug+f'{enc}_distill_{distill_loss_type}')
    norm=fold==1
    model=RegressionPLModel(
        enc='r50',
        pred_shape=pred_shape,
        size=lr_size,
        total_steps=len(train_loader),
        teacher_ckpt=teacher_checkpoint,
        distill_alpha=distill_alpha,
        student_in_chans=lr_in_chans,
        teacher_in_chans=hr_in_chans,
        distill_loss_type=distill_loss_type,
        infonce_temperature=infonce_temperature
    )
    
    state_dict=torch.load('./r3d50_KM_200ep.pth')["state_dict"]
    conv1_weight = state_dict['conv1.weight']
    state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    model.backbone.load_state_dict(state_dict,strict=False)
    
    print('FOLD : ',fold)
    print(f'DISTILLATION LOSS TYPE: {distill_loss_type}')
    if distill_loss_type == 'infonce':
        print(f'InfoNCE Temperature: {infonce_temperature}')
    wandb_logger.watch(model, log="all", log_freq=100)
    multiplicative = lambda epoch: 0.9

    trainer = pl.Trainer(
        max_epochs=15,
        accelerator="gpu",
        devices=1,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=32,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='auto',
        callbacks=[ModelCheckpoint(filename=f'distill_2um_64layers_{fid}_{fold}_fr_{enc}'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),

                    ],

    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()