import os.path as osp
import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 1099511627776
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = '1099511627776'
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import timm
import random
import yaml
import glob
import threading
import numpy as np
import pandas as pd
from data import *
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
from skimage.transform import downscale_local_mean
from einops import rearrange
from timesformer_pytorch_model import TimeSformer

PIL.Image.MAX_IMAGE_PIXELS = 933120000



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
    in_chans = 8 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 518
    tile_size = 518
    stride = tile_size // 7

    train_batch_size = 2 # 32
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

        A.OneOf([
        # A.GaussNoise(var_limit=[10, 50]),
        A.RandomBrightnessContrast(p=1),
        # A.CLAHE(p=1),
        ], p=.75),
        A.ShiftScaleRotate(rotate_limit=360,shift_limit=0.15,scale_limit=0.1,p=0.75),
        A.OneOf([
                # A.GaussNoise(var_limit=[10, 50]),
                A.GaussianBlur(),
                A.MotionBlur(),
                ], p=0.4),
        # A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
        A.CoarseDropout(max_holes=2, max_width=int(size * 0.2), max_height=int(size * 0.2), 
                        mask_fill_value=0, p=0.5),
        # A.Cutout(max_h_size=int(size * 0.6),
        #          max_w_size=int(size * 0.6), num_holes=1, p=1.0),
        A.Normalize(
            mean= [0.4560]*in_chans,
            std= [0.2250]*in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]

    valid_aug_list = [
        A.Resize(size, size),
        A.Normalize(
            mean= [0.4560]*in_chans,
            std= [0.2250]*in_chans
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

def read_image_mask(fragment_id,start_idx=24,end_idx=42, CFG=CFG):
    fragment_id_ = fragment_id.split("_")[0]
    images = []
    idxs = range(start_idx, end_idx)
    if 'unroll' in fragment_id:
        idxs = range(1, 19)
    for i in idxs:
        if os.path.exists(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif"):
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.tif", 0)
        elif 'rag' in fragment_id:
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.png", 0)

        else:
            image = cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/layers/{i:02}.jpg", 0)
        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)        
        # image=np.clip(image,0,200)
        images.append(image)
    images = np.stack(images, axis=2)
    if any(id_ in fragment_id_ for id_ in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']):
        images=images[:,:,::-1]
    # Get the list of files that match the pattern
    inklabel_files = glob.glob(f"train_scrolls/{fragment_id}/*inklabels*")
    if len(inklabel_files) > 0:
        mask = cv2.imread( inklabel_files[0], 0)
    else:
        print(f"Creating empty mask for {fragment_id}")
        mask = np.zeroes(images[0].shape)
    try:
        fragment_mask=cv2.imread(glob.glob(f'train_scrolls/{fragment_id}/*mask.png')[0], 0)
        # fragment_mask=cv2.imread(CFG.comp_dataset_path + f"train_scrolls/{fragment_id}/{fragment_id_}_mask.png", 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    except:
        fragment_mask=np.ones((images.shape[0],images.shape[1]))
    if 'rag' in fragment_id or fragment_id in ['20240501145915','20231130150016']:
        
        mask = np.pad(mask, [(0, pad0), (0, pad1)], constant_values=0)
    # images=downscale_local_mean(images, (1, 1,2))
    mask = mask.astype('float32')
    mask/=255
    print(fragment_id,images.shape,mask.shape)
    return images, mask,fragment_mask

def worker_function(fragment_id, CFG):
    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []


    # try:
# for fragment_id in ['20250807020208','20250806145318','l2_0','auto_grown_20250926162450723','auto_grown_20250919055754487_inp_hr','david_9b','658','500p2a','-1']:
#,
        
    # for fragment_id in ['20230522181603','20230702185752','20230827161847','20230909121925','20230905134255','20230904135535']:
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
    elif 'Frag' in fragment_id or fragment_id in ['20241108120732','20231016151000']:
        segment=Segment(                
            segment_id=fragment_id,
                layer_range=(24,40),
                reverse_layers=False,
                base_path='train_scrolls',
                tile_size=CFG.tile_size,
                # xy_scale=.5,
                       )
    elif '08312025' in fragment_id:
        segment=Segment(                
                    segment_id=fragment_id,
                        layer_range=(8,24),
                        reverse_layers=False,
                        base_path='0139_traces',
                        tile_size=CFG.tile_size,
                        # xyz_scale=1.46,
                               )
    elif fragment_id in ['500p2a','658'] :
        segment=Segment(                
            segment_id=fragment_id,
                layer_range=(1,63),
                reverse_layers=True,
                base_path='train_scrolls',
                tile_size=CFG.tile_size,
                xyz_scale=.258,
                       )
    elif fragment_id in ['auto_grown_20250919055754487_inp_hr','david_9b'] :
        segment=Segment(                
            segment_id=fragment_id,
                layer_range=(1,63),
                reverse_layers=False,
                base_path='new_traces9b',
                tile_size=CFG.tile_size,
                xyz_scale=.258,
                       )
    elif fragment_id in ['auto_grown_20250926162450723'] :
        segment=Segment(                
            segment_id=fragment_id,
                layer_range=(1,63),
                reverse_layers=False,
                base_path='841',
                tile_size=CFG.tile_size,
                xyz_scale=.258,
                       )
    elif fragment_id in ['l2_0'] :
        segment=Segment(                
            segment_id=fragment_id,
                layer_range=(1,63),
                reverse_layers=False,
                base_path='0139_columns',
                tile_size=CFG.tile_size,
                xyz_scale=.258,
                       )
    elif fragment_id in ['-1','-2'] :
        segment=Segment(                
            segment_id=fragment_id,
                layer_range=(1,63),
                reverse_layers=False,
                base_path='sean_hiddenlayers',
                tile_size=CFG.tile_size,
                xyz_scale=.258,
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
                base_path='0175_2um',
                tile_size=CFG.tile_size,
                xyz_scale=.258,
                       )
    image, mask,fragment_mask = segment.get_data()
    # image, mask, fragment_mask = read_image_mask(fragment_id, CFG=CFG)
    # except Exception as e:
    #     print("aborted reading fragment", fragment_id,e)
    image=downscale_local_mean(image, (1, 1,2))
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    windows_dict={}

    for a in y1_list:
        for b in x1_list:
            if not np.any(fragment_mask[a:a + CFG.tile_size, b:b + CFG.tile_size]==0):
                if (fragment_id==CFG.valid_id) or (not np.all(mask[a:a + CFG.tile_size, b:b + CFG.tile_size]<0.05)):
                    for yi in range(0,CFG.tile_size,CFG.size):
                        for xi in range(0,CFG.tile_size,CFG.size):
                            y1=a+yi
                            x1=b+xi
                            y2=y1+CFG.size
                            x2=x1+CFG.size
                            if (y1,y2,x1,x2) not in windows_dict:
                                if fragment_id!=CFG.valid_id:
                                    if image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans) and mask[y1:y2, x1:x2].shape==(CFG.size,CFG.size):
                                        train_images.append(image[y1:y2, x1:x2])
                                        train_masks.append(1-mask[y1:y2, x1:x2, None])

                                    
                                    windows_dict[(y1,y2,x1,x2)]='1'

                            if fragment_id==CFG.valid_id:
                                if (y1,y2,x1,x2) not in windows_dict:
                                    if image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans) and mask[y1:y2, x1:x2].shape==(CFG.size,CFG.size):
                                        valid_images.append(image[y1:y2, x1:x2])
                                        valid_masks.append(1-mask[y1:y2, x1:x2, None])
                                        valid_xyxys.append([x1, y1, x2, y2])
                                        # assert image[y1:y2, x1:x2].shape==(CFG.size,CFG.size,CFG.in_chans)
                                        windows_dict[(y1,y2,x1,x2)]='1'

    print("finished reading fragment", fragment_id,image.shape,mask.shape,fragment_mask.shape)

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys
# '20241025145341','20241025145701','20241025150211','20241108111522','20241108115232','20241108120732','20241113070770','20241113080880','20241113090990','20241030152031','20231210121321','Frag2','Frag3','Frag1','20240501145915','20231130150016','20231012184423'
# '20241025145341','20241025145701','20241025150211','20241030152031','20241108111522','20241108115232','20241108120732','20241113070770','20241113080880','20241113090990','working_mesh_0_window_445437_495437_flatboi_2'
# 'frag1','frag2','frag3','frag4','frag6','frag5','20240618142022','20241015063650','20240917165839','20240917165329','20240917093139','20240917092608'
def get_train_valid_dataset(fragment_ids=['20241108120732','20241113070770','Frag2','Frag3','Frag1','20250807020208','20250806145318','l2_0','auto_grown_20250926162450723','auto_grown_20250919055754487_inp_hr','david_9b','658','500p2a','-1']):
    threads = []
    results = [None] * len(fragment_ids)

    # Function to run in each thread
    def thread_target(idx, fragment_id):
        results[idx] = worker_function(fragment_id, CFG)

    # Create and start threads
    for idx, fragment_id in enumerate(fragment_ids):
        thread = threading.Thread(target=thread_target, args=(idx, fragment_id))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    train_images = []
    train_masks = []
    valid_images = []
    valid_masks = []
    valid_xyxys = []
    print("Aggregating results")
    for r in results:
        if r is None:
            continue
        train_images += r[0]
        train_masks += r[1]
        valid_images += r[2]
        valid_masks += r[3]
        valid_xyxys += r[4]

    return train_images, train_masks, valid_images, valid_masks, valid_xyxys

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    return aug

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        
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
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            xy=self.xyxys[idx]
            if self.transform:
                # image=np.max(image,axis=2)
                # image=np.stack([np.max(image,axis=2),np.median(image,axis=2).astype(np.uint8),np.min(image[:,:,8:15],axis=2)],axis=2)
                data = self.transform(image=image, mask=label)
                image = data['image']
                # image=image.unsqueeze(0)
                image=torch.stack([image,image,image],dim=0)
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//14,self.cfg.size//14)).squeeze(0)
            return image, label,xy
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
            # print(image.shape)
            # image=self.fourth_augment(image)
            
            if self.transform:
                pct_offset= np.random.uniform(0.75, 1.0)
                start=np.random.randint(0, 8)
                # image=np.max(image,axis=2)
                # image=np.stack([np.max(image,axis=2),np.median(image,axis=2).astype(np.uint8),np.min(image[:,:,8:15],axis=2)],axis=2)
                # print(image.shape)
                data = self.transform(image=image, mask=label)
                image = data['image']
                # image=image.unsqueeze(0)
                image=torch.stack([image,image,image],dim=0)
                
                label = data['mask']
                label=F.interpolate(label.unsqueeze(0),(self.cfg.size//14,self.cfg.size//14)).squeeze(0)
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

model_name='vit_base_patch14_reg4_dinov2'
model=timm.create_model(model_name, pretrained=True)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

class RegressionPLModel(pl.LightningModule):
    def __init__(self,pred_shape,size=256,enc='',with_norm=False,total_steps=780):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2= smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func= lambda x,y:0.5 * self.loss_func1(x,y)+0.5*self.loss_func2(x,y)
        self.model_name='vit_base_patch14_reg4_dinov2'
        self.backbone=timm.create_model(model_name, pretrained=True)
        # for p in self.backbone.parameters():
        #     p.requires_grad=False
        self.linear=torch.nn.Linear(512,1)
        self.depth=CFG.in_chans
        self.timesformer=TimeSformer(
                dim = 512,
                image_size = 37,
                patch_size = 1,
                num_frames = self.depth,
                num_classes = 16,
                channels=768,
                depth = 4,
                heads = 8,
                dim_head =  64,
                attn_dropout = 0.1,
                ff_dropout = 0.1
            )
    def forward(self,x):
        x=rearrange(x, 'b c d h w -> (b d) c h w')
        # with torch.no_grad():
        x=self.backbone.forward_features(x)
        x=rearrange(x[:,5:], '(b d) (h w) e -> b d e h w ',d=self.depth,h=37,w=37)
        x=self.timesformer.forward_features(x)
        x=rearrange(x[:,1:,:],'b (f h w) d -> b f h w d',f=self.depth,h=37,w=37)
        x=x.max(dim=1)[0]
        return self.linear(x).squeeze(-1).unsqueeze(1)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        outputs = self(x)
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
            self.mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=14,mode='bilinear').squeeze(0).squeeze(0).numpy()
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

        optimizer = AdamW([p for p in self.parameters() if p.requires_grad], lr=CFG.lr)
        scheduler =torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-4,pct_start=0.1, steps_per_epoch=self.hparams.total_steps, epochs=3,final_div_factor=1e2)
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

fragments=['20250807020208']
enc_i,enc,fold=0,'i3d',0
for fid in fragments:
    CFG.valid_id=fid
    fragment_id = CFG.valid_id
    run_slug=f'training_scrolls_dinotimes_valid={fragment_id}_{CFG.size}x{CFG.size}_submissionlabels_wild11'

    valid_mask_gt = cv2.imread(CFG.comp_dataset_path + f"0175_2um/{fragment_id}/{fragment_id}_inklabels.png", 0)
    valid_mask_gt=cv2.resize(valid_mask_gt, (0,0), fx=0.258, fy=0.258) 

    pred_shape=valid_mask_gt.shape

    train_images, train_masks, valid_images, valid_masks, valid_xyxys = get_train_valid_dataset()
    print(len(train_images))
    valid_xyxys = np.stack(valid_xyxys)
    train_dataset = CustomDataset(
        train_images, CFG, labels=train_masks, transform=get_transforms(data='train', cfg=CFG))
    valid_dataset = CustomDataset(
        valid_images, CFG,xyxys=valid_xyxys, labels=valid_masks, transform=get_transforms(data='valid', cfg=CFG))

    train_loader = DataLoader(train_dataset,
                                batch_size=CFG.train_batch_size,
                                shuffle=True,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True,
                                )
    valid_loader = DataLoader(valid_dataset,
                                batch_size=CFG.valid_batch_size,
                                shuffle=False,
                                num_workers=CFG.num_workers, pin_memory=True, drop_last=True)

    wandb_logger = WandbLogger(project="vesivus",name=run_slug+f'{enc}_finetune')
    norm=fold==1
    model2=RegressionPLModel.load_from_checkpoint(CFG.model_dir+'dinotimes_base_frozen_allnewdata20250807020208_0_fr_i3depoch=20.ckpt',enc='i3d',pred_shape=pred_shape,size=CFG.size,total_steps=len(train_loader))
    model=RegressionPLModel(enc='i3d',pred_shape=pred_shape,size=CFG.size,total_steps=len(train_loader))
    model.load_state_dict(model2.state_dict())

    print('FOLD : ',fold)
    wandb_logger.watch(model, log="all", log_freq=100)
    multiplicative = lambda epoch: 0.9

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        # check_val_every_n_epoch=4,
        logger=wandb_logger,
        default_root_dir="./models",
        accumulate_grad_batches=32,
        precision='16-mixed',
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        strategy='auto',
        callbacks=[ModelCheckpoint(filename=f'dinotimes_base_ft_allnewdata{fid}_{fold}_fr_{enc}'+'{epoch}',dirpath=CFG.model_dir,monitor='train/total_loss',mode='min',save_top_k=CFG.epochs),

                    ],

    )
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    wandb.finish()