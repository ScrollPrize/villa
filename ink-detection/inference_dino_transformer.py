import os.path as osp
import os
import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
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
import scipy.stats as st

PIL.Image.MAX_IMAGE_PIXELS = 933120000
from data import *


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
    in_chans = 9 # 65
    encoder_depth=5
    # ============== training cfg =============
    size = 518
    tile_size = 518
    stride = tile_size // 4

    train_batch_size = 16 # 32
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
        if os.path.exists(CFG.comp_dataset_path + f"0139_traces/{fragment_id}/layers/{i:02}.tif"):
            image = cv2.imread(CFG.comp_dataset_path + f"0139_traces/{fragment_id}/layers/{i:02}.tif", 0)
        # elif 'rag' in fragment_id:
        #     image = cv2.imread(CFG.comp_dataset_path + f"0139_traces/{fragment_id}/layers/{i:02}.png", 0)

        else:
            image = cv2.imread(CFG.comp_dataset_path + f"0139_traces/{fragment_id}/layers/{i:02}.jpg", 0)
        pad0 = (256 - image.shape[0] % 256)
        pad1 = (256 - image.shape[1] % 256)
        image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)        
        # image=np.clip(image,0,200)
        # image = cv2.resize(image, (image.shape[1]*2,image.shape[0]*2), interpolation = cv2.INTER_AREA)
        images.append(image)
        
    images = np.stack(images, axis=2)
    if any(id_ in fragment_id_ for id_ in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']):
        images=images[:,:,::-1]
    # Get the list of files that match the pattern
    try:
        fragment_mask=cv2.imread(glob.glob(f'0139_traces/{fragment_id}/*mask*')[0], 0)
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
    except:
        fragment_mask=np.ones((images.shape[0],images.shape[1]))
    images=downscale_local_mean(images, (1, 1,2))

    return images,fragment_mask
# def read_image_mask(fragment_id,start_idx=18,end_idx=38,rotation=0):

#     images = []

#     # idxs = range(65)
#     mid = 65 // 2
#     start = mid - CFG.in_chans // 2
#     end = mid + CFG.in_chans // 2
#     idxs = range(start_idx, end_idx)
#     # idxs = range(0, 65)

#     for i in idxs:
        
#         image = cv2.imread(f"./0139_traces/{fragment_id}/layers/{i:02}.tif", 0)

#         pad0 = (256 - image.shape[0] % 256)
#         pad1 = (256 - image.shape[1] % 256)

#         image = np.pad(image, [(0, pad0), (0, pad1)], constant_values=0)
#         # image = ndimage.median_filter(image, size=5)
#         image=np.clip(image,0,200)
#         # image = cv2.flip(image, 0)

#         images.append(image)
#     images = np.stack(images, axis=2)
#     if fragment_id in ['20230701020044','verso','20230901184804','20230901234823','20230531193658','20231007101615','20231005123333','20231011144857','20230522215721', '20230919113918', '20230625171244','20231022170900','20231012173610','20231016151000']:
#         images=images[:,:,::-1]

#     fragment_mask=None
#     if os.path.exists(f'./0139_traces/{fragment_id}/{fragment_id}_mask.png'):
#         fragment_mask=cv2.imread(CFG.comp_dataset_path + f"0139_traces/{fragment_id}/{fragment_id}_mask.png", 0)
#         fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)

#     return images,fragment_mask

def get_img_splits(fragment_id,s,e,rotation=0):
    images = []
    xyxys = []
    if fragment_id in os.listdir('train_scrolls'):
        segment=Segment(                
                segment_id=fragment_id,
                    layer_range=(24,40),
                    reverse_layers=False,
                    base_path='train_scrolls',
                    tile_size=CFG.tile_size,
                    # xyz_scale=.5,
    
                           )
    # elif fragment_id in os.listdir('841'):
    #     segment=Segment(                
    #             segment_id=fragment_id,
    #                 layer_range=(0,62),
    #                 reverse_layers=False,
    #                 base_path='841',
    #                 tile_size=CFG.tile_size,
    #                 xyz_scale=.258,
    
    #                        )
    elif fragment_id in os.listdir('0175_2um'):
        segment=Segment(                
                segment_id=fragment_id,
                    layer_range=(0,64),
                    reverse_layers=False,
                    base_path='0175_2um',
                    tile_size=CFG.tile_size,
                    xyz_scale=.25,
                           ) 
    else:
        segment=Segment(                
                segment_id=fragment_id,
                    layer_range=(7,25) if os.path.exists(f'841_9um/{fragment_id}/layers/24.tif') else (2,18),
                    reverse_layers=False,
                    base_path='841_9um',
                    tile_size=CFG.tile_size,
                    # z_scale=.5,
                           )
        
    image, _,fragment_mask = segment.get_data()
    image=downscale_local_mean(image, (1, 1,2))

    print(image.shape)
    x1_list = list(range(0, image.shape[1]-CFG.tile_size+1, CFG.stride))
    y1_list = list(range(0, image.shape[0]-CFG.tile_size+1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            if not np.any(fragment_mask[y1:y2, x1:x2]==0):
                images.append(image[y1:y2, x1:x2])
                xyxys.append([x1, y1, x2, y2])
    test_dataset = CustomDatasetTest(images,np.stack(xyxys), CFG,transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(
            mean= [0.4560] * CFG.in_chans,
            std= [0.2250] * CFG.in_chans
        ),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                              batch_size=CFG.valid_batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False,
                              )
    return test_loader, np.stack(xyxys),(image.shape[0],image.shape[1]),fragment_mask

def get_transforms(data, cfg):
    if data == 'train':
        aug = A.Compose(cfg.train_aug_list)
    elif data == 'valid':
        aug = A.Compose(cfg.valid_aug_list)

    # print(aug)
    return aug
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class CustomDataset(Dataset):
    def __init__(self, images ,cfg,xyxys=None, labels=None, transform=None):
        self.images = images
        self.cfg = cfg
        self.labels = labels
        self.transform = transform
        self.xyxys=xyxys
        self.kernel=gkern(64,2)
        self.kernel/=self.kernel.max()
        self.kernel=torch.FloatTensor(self.kernel)
    def __len__(self):
        # return len(self.df)
        return len(self.images)

    def __getitem__(self, idx):
        if self.xyxys is not None:
            image = self.images[idx]
            label = self.labels[idx]
            offset=4
            image=image[:,:,offset:offset+self.cfg.in_chans]
            xy=self.xyxys[idx]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                # label= torch.mul(self.kernel,data['mask'])
                label = label.mean().type(torch.float32)

            return image, label,xy
        else:
            image = self.images[idx]
            label = self.labels[idx]
            # offset=random.choice([0,1,2,3,4])
            offset=4
            image=image[:,:,offset:offset+self.cfg.in_chans]
            if self.transform:
                data = self.transform(image=image, mask=label)
                image = data['image'].unsqueeze(0)
                label= torch.mul(self.kernel,data['mask'])
                label = label.mean().type(torch.float32)
            
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
            # image=np.max(image,axis=2)
            # image=np.stack([np.max(image,axis=2),np.median(image,axis=2).astype(np.uint8),np.min(image[:,:,12:18],axis=2)],axis=2)
            data = self.transform(image=image)
            image = data['image']
            # image=image.unsqueeze(0)
            image=torch.stack([image,image,image],dim=0)

        return image,xy

    






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
        # for l in self.convs:
        #     for m in l._modules:
        #         init_weights(m)
    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask



from collections import OrderedDict




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
        self.backbone=timm.create_model(self.model_name, pretrained=True)
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
        self.log("train/Arcface_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
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

        self.log("val/MSE_loss", loss1.item(),on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss1}
    
    def on_validation_epoch_end(self):
        self.mask_pred = np.divide(self.mask_pred, self.mask_count, out=np.zeros_like(self.mask_pred), where=self.mask_count!=0)
        wandb_logger.log_image(key="masks", images=[np.clip(self.mask_pred,0,1)], caption=["probs"])

        #reset mask
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)
    def configure_optimizers(self):
        optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=CFG.lr)    
        scheduler = get_scheduler(CFG, optimizer)
        return [optimizer]
import torch.nn as nn
import torch
import math
import time
import numpy as np
import torch

from warmup_scheduler import GradualWarmupScheduler


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
        optimizer, cfg.epochs, eta_min=1e-7)
    scheduler = GradualWarmupSchedulerV2(
        optimizer, multiplier=10, total_epoch=1, after_scheduler=scheduler_cosine)

    return scheduler

def scheduler_step(scheduler, avg_val_loss, epoch):
    scheduler.step(epoch)
   
import torch as tc
def TTA(x:tc.Tensor,model:nn.Module):
    #x.shape=(batch,c,h,w)
    shape=x.shape
    x=[x,*[tc.rot90(x,k=i,dims=(-2,-1)) for i in range(1,4)],]
    x=tc.cat(x,dim=0)
    x=model(x)
    # x=torch.sigmoid(x)
    # print(x.shape)
    x=x.reshape(4,shape[0],CFG.size//4,CFG.size//4)
    
    x=[tc.rot90(x[i],k=-i,dims=(-2,-1)) for i in range(4)]
    x=tc.stack(x,dim=0)
    return x.mean(0)

DiceLoss = smp.losses.DiceLoss(mode='binary')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
FocalLoss=smp.losses.FocalLoss(gamma=2.5,mode='binary')
alpha = 0.5
beta = 1 - alpha
TverskyLoss = smp.losses.TverskyLoss(
    mode='binary', log_loss=False, alpha=alpha, beta=beta)
MSELoss=nn.MSELoss()
HuberLoss=nn.HuberLoss(delta=5.0)
NNBCE=nn.BCEWithLogitsLoss(pos_weight=torch.ones([1]).to('cuda')*5 )
def criterion(y_pred, y_true):
    # return 0.5 * BCELoss(y_pred, y_true) + 0.5 * DiceLoss(y_pred, y_true)
    return HuberLoss(y_pred, y_true)
def normalization(x):
    """input.shape=(batch,f1,f2,...)"""
    #[batch,f1,f2]->dim[1,2]
    dim=list(range(1,x.ndim))
    mean=x.mean(dim=dim,keepdim=True)
    std=x.std(dim=dim,keepdim=True)
    return (x-mean)/(std+1e-9)
# def predict_fn(test_loader, model, device, test_xyxys,pred_shape):
#     mask_pred = np.zeros((pred_shape[0]//14,pred_shape[1]//14))
#     mask_count = np.zeros((pred_shape[0]//14,pred_shape[1]//14))
#     kernel=gkern(CFG.size,1)
#     kernel=kernel/kernel.max()
#     model.eval()

#     for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
#         images = images.to(device)
#         batch_size = images.size(0)
#         with torch.no_grad():
#             with torch.autocast(device_type="cuda"):
#                 y_preds = model(images)
#             # y_preds =TTA(images,model)
#         # y_preds = y_preds.to('cpu').numpy()

#         y_preds = torch.sigmoid(y_preds).squeeze(1).to('cpu')
#         for i, (x1, y1, x2, y2) in enumerate(xys):
#             x1, y1, x2, y2=x1//14, y1//14, x2//14, y2//14
#             mask_pred[y1:y2, x1:x2] += y_preds[i].numpy()
#             # mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
#             mask_count[y1:y2, x1:x2] += np.ones((CFG.size//14, CFG.size//14))

#     mask_pred /= mask_count
#     # mask_pred/=mask_pred.max()
#     return mask_pred
    # return losses.avg,
def predict_fn(test_loader, model, device, test_xyxys,pred_shape):
    mask_pred = np.zeros(pred_shape)
    mask_count = np.zeros(pred_shape)
    kernel=gkern(CFG.size,1)
    kernel=kernel/kernel.max()
    model.eval()

    for step, (images,xys) in tqdm(enumerate(test_loader),total=len(test_loader)):
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
            # y_preds =TTA(images,model)
        # y_preds = y_preds.to('cpu').numpy()

        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(F.interpolate(y_preds[i].unsqueeze(0).float(),scale_factor=14,mode='bilinear').squeeze(0).squeeze(0).numpy(),kernel)
            # mask_pred[y1:y2, x1:x2] += F.interpolate(y_preds[i].unsqueeze(0).unsqueeze(0).float(),scale_factor=4,mode='bilinear').squeeze(0).squeeze(0).numpy()
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    # mask_pred/=mask_pred.max()
    return mask_pred
#     # return losses.avg,[]
from PIL import Image
from PIL.ImageOps import equalize,autocontrast
import gc
import time


for m in ['checkpoints/dinotimes_base_finetune_alldata20241113080880_0_fr_i3depoch=2.ckpt']:
    # model=torch.jit.load(f'models_norm/{m}')
    model=RegressionPLModel.load_from_checkpoint(m,strict=False,enc='resnest101')
    model.cuda()
    model.eval()
    wandb.init(
      # Set the project where this run will be logged
      project="vesivus", 
      # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
      name=f"ALL_scrolls_tta_{m}", 
      # Track hyperparameters and run metadata
        )
# '20231210132040','20240304141531','20240304144031','20240304161941','working_mesh_flatboi_2','working_mesh_flatboi_3','working_mesh_flatboi_4','2','3','05201425','20240618142020','20240716140050','20240712064330','20240712071520','20240712074250','20240715203740','seg4','seg5','seg6','seg7','seg8','20240521175507','20240521175602','20240521175659','20240227180741','20240228135119','20240228191147','20240303035326','20240309033932'
# '2','3','05201425','20240618142020','20240716140050','20240712064330','20240712071520','20240712074250','20240715203740','seg4','seg5','seg6','seg7','seg8','20240521175507','20240521175602','20240521175659','20240227180741','20240228135119','20240228191147','20240303035326','20240309033932'
    # '20231210132040','20231005123336','20240304141531','20240304144031','20240304161941','working_mesh_0_window_445437_495437_flatboi','working_mesh_flatboi_4','2','3','05201425','20240618142020','20240716140050','20240712064330','20240712071520','20240712074250','20240715203740','seg4','seg5','seg6','seg7','seg8','20240521175507','20240521175602','20240521175659','20240227180741','20240228135119','20240228191147','20240303035326','20240309033932','20241017164948','20241017165012','20241017165105','20241023103934','20241023104051','20241023104311','20241023104551','20241023104629','Frag1','Frag2','Frag3','Frag4','Frag5'
    # ['frag5','20240917091538','20240917091756','20240917092034','20240917092608','20240917093139','20240917164959','20240917165329','20240917165839','20240917170131','20240917170315','20240917170604','20240813112037','20240818211653','20240816164032','20240820102547','20240815194445','20240814124338','20240814122007','20240820133348','20240820150159']
    # for fragment_id in ['s5_inner_wrap']:
    for fragment_id in os.listdir('841_9um'):
        if True:
            preds=[]
            for r in [0]:
                for i in [6]:
                    try:
                        start_f=i
                        end_f=start_f+2*CFG.in_chans
                        test_loader,test_xyxz,test_shape,fragment_mask=get_img_splits(fragment_id,start_f,end_f,r)
                        mask_pred= predict_fn(test_loader, model, device, test_xyxz,test_shape)
                        mask_pred=np.clip(np.nan_to_num(mask_pred),a_min=0,a_max=1)
                        mask_pred/=mask_pred.max()
    
                        preds.append(mask_pred)
    
                        img=wandb.Image(
                        mask_pred, 
                        caption=f"{fragment_id}"
                        )
                        wandb.log({'predictions':img})
            
                        # print("plot time: ",t5-t4)
                        gc.collect()
                    except Exception as e:
                        print(e)
                        continue
        
    del mask_pred,test_loader,model
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
