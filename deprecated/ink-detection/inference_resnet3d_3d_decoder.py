import os.path as osp
import os
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2,40))
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import wandb

import random
import sys
import yaml

import numpy as np
import pandas as pd

import gc
import tempfile
import pytorch_lightning as pl
import numpy as np
import scipy.stats as st

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

from torch.utils.data import DataLoader
import cv2
import segmentation_models_pytorch as smp
from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models.resnetall import generate_model
from train_resnet3d_lib.script_utils import resolve_fragment_base_path
from train_resnet3d_lib.legacy_segment_adapter import load_inference_segment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class CFG:
    comp_name = 'vesuvius'
    comp_dir_path = './'
    comp_folder_name = './'
    comp_dataset_path = f'./'
    exp_name = 'pretraining_all_3d_decoder'

    target_size = 1
    model_name = 'Unet'
    backbone = 'resnet3d'
    in_chans = 62
    encoder_depth = 5

    size = 256
    tile_size = 256
    stride = tile_size // 5

    train_batch_size = 32
    valid_batch_size = 32
    use_amp = True
    epochs = 50
    lr = 1e-4 / 10
    min_lr = 1e-6
    weight_decay = 1e-6
    max_grad_norm = 5
    print_freq = 50
    num_workers = 4
    seed = 42

    print('set dataset path')
    outputs_path = f'./outputs/{comp_name}/{exp_name}/'
    model_dir = outputs_path + f'{comp_name}-models/'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_INFERENCE_BASE_PATHS = (
    '1451_2um',
    's4_2um',
    'PherMANBp_2um',
    '841_2um',
    '814_2um',
    'Man5',
)


def get_img_splits(fragment_id, reverse_layers=False, base_path=None):
    """Load a segment and tile it for inference."""
    images = []
    xyxys = []

    if base_path is None:
        base_path = resolve_fragment_base_path(
            fragment_id,
            DEFAULT_INFERENCE_BASE_PATHS,
            fallback_root='2um_dataset',
        )

    image, fragment_mask = load_inference_segment(
        segment_id=fragment_id,
        dataset_root=base_path,
        layer_range=(1, 63),
        reverse_layers=reverse_layers,
        in_chans=CFG.in_chans,
        layer_read_workers=CFG.num_workers,
    )

    # For large segments, back image with a disk memmap to avoid bloating Python's heap
    image_gb = image.nbytes / 1e9
    memmap_path = None
    if image_gb > 50:
        print(f'  Large image ({image_gb:.1f} GB) — backing with memmap', flush=True)
        memmap_path = tempfile.mktemp(suffix='.dat', prefix='vesuvius_img_')
        mm = np.memmap(memmap_path, dtype=image.dtype, mode='w+', shape=image.shape)
        mm[:] = image
        mm.flush()
        del image
        gc.collect()
        image = mm

    x1_list = list(range(0, image.shape[1] - CFG.tile_size + 1, CFG.stride))
    y1_list = list(range(0, image.shape[0] - CFG.tile_size + 1, CFG.stride))
    for y1 in y1_list:
        for x1 in x1_list:
            y2 = y1 + CFG.tile_size
            x2 = x1 + CFG.tile_size
            # Quick check: skip tiles where every pixel across all channels is zero
            if not np.all(image[y1:y2, x1:x2] == 0):
                xyxys.append([x1, y1, x2, y2])
    xyxys_arr = np.stack(xyxys)
    print(f'  {len(xyxys)} tiles to infer ({image.shape})', flush=True)

    # Reduce workers for large segments to avoid fork memory overhead
    num_workers = 2 if image_gb > 50 else CFG.num_workers

    test_dataset = LazyTileDataset(image, xyxys_arr, CFG, transform=A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(mean=[0] * CFG.in_chans, std=[1] * CFG.in_chans),
        ToTensorV2(transpose_mask=True),
    ]))

    test_loader = DataLoader(test_dataset,
                             batch_size=CFG.valid_batch_size,
                             shuffle=False,
                             num_workers=num_workers, pin_memory=True, drop_last=False)
    return test_loader, xyxys_arr, (image.shape[0], image.shape[1]), fragment_mask, memmap_path


class LazyTileDataset(Dataset):
    """Reads tiles lazily from the full image array — no per-tile RAM copies."""
    def __init__(self, image, xyxys, cfg, transform=None):
        self.image = image      # shared reference, not copied
        self.xyxys = xyxys
        self.cfg = cfg
        self.transform = transform

    def __len__(self):
        return len(self.xyxys)

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.xyxys[idx]
        tile = self.image[y1:y2, x1:x2].copy()  # small copy at read time
        if self.transform:
            data = self.transform(image=tile)
            tile = data['image'].unsqueeze(0)
        return tile, self.xyxys[idx]


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
        attn = self.attn_conv(x)
        attn = F.softmax(attn, dim=2)
        out = (x * attn).sum(dim=2)
        return out


class AuxHead(nn.Module):
    """Auxiliary prediction head: adaptive avg pool depth -> Conv2d 1x1 -> bilinear upsample."""
    def __init__(self, in_ch, target_size=64):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.conv = nn.Conv2d(in_ch, 1, 1)
        self.target_size = target_size

    def forward(self, x):
        x = self.pool(x).squeeze(2)
        x = self.conv(x)
        if x.shape[-1] != self.target_size:
            x = F.interpolate(x, size=(self.target_size, self.target_size),
                              mode='bilinear', align_corners=False)
        return x


class Decoder3DUNet(nn.Module):
    """3D UNet decoder with channel reduction, skip connections, depth attention collapse."""
    def __init__(self, encoder_dims=(256, 512, 1024, 2048),
                 decoder_dims=(64, 128, 256, 512), deep_supervision=True):
        super().__init__()
        self.deep_supervision = deep_supervision

        self.channel_reduce = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(enc_d, dec_d, 1, bias=False),
                nn.GroupNorm(min(32, dec_d), dec_d),
                nn.ReLU(inplace=True)
            ) for enc_d, dec_d in zip(encoder_dims, decoder_dims)
        ])

        self.decoder_blocks = nn.ModuleList([
            ResConvBlock3D(decoder_dims[i] + decoder_dims[i - 1], decoder_dims[i - 1])
            for i in range(len(decoder_dims) - 1, 0, -1)
        ])

        self.depth_collapse = DepthAttentionCollapse(decoder_dims[0])
        self.logit = nn.Conv2d(decoder_dims[0], 1, 1)

        if deep_supervision:
            self.aux_head_s2 = AuxHead(decoder_dims[2])
            self.aux_head_s1 = AuxHead(decoder_dims[1])

    def forward(self, feat_maps):
        feats = [self.channel_reduce[i](feat_maps[i]) for i in range(4)]
        aux_outputs = []

        x = feats[3]
        x = F.interpolate(x, size=feats[2].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, feats[2]], dim=1)
        x = self.decoder_blocks[0](x)
        if self.deep_supervision and self.training:
            aux_outputs.append(self.aux_head_s2(x))

        x = F.interpolate(x, size=feats[1].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, feats[1]], dim=1)
        x = self.decoder_blocks[1](x)
        if self.deep_supervision and self.training:
            aux_outputs.append(self.aux_head_s1(x))

        x = F.interpolate(x, size=feats[0].shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x, feats[0]], dim=1)
        x = self.decoder_blocks[2](x)

        x = self.depth_collapse(x)
        x = self.logit(x)

        if self.deep_supervision and self.training:
            return x, aux_outputs
        return x


# ======================== Lightning Model ========================

class RegressionPLModel(pl.LightningModule):
    def __init__(self, pred_shape, size=256, enc='', with_norm=False, total_steps=780):
        super(RegressionPLModel, self).__init__()

        self.save_hyperparameters()
        self.mask_pred = np.zeros(self.hparams.pred_shape)
        self.mask_count = np.zeros(self.hparams.pred_shape)

        self.loss_func1 = smp.losses.DiceLoss(mode='binary')
        self.loss_func2 = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25)
        self.loss_func = lambda x, y: 0.5 * self.loss_func1(x, y) + 0.5 * self.loss_func2(x, y)

        self.backbone = generate_model(model_depth=152, n_input_channels=1, forward_features=True, n_classes=1039)

        self.decoder = Decoder3DUNet(
            encoder_dims=(256, 512, 1024, 2048),
            decoder_dims=(64, 128, 256, 512),
            deep_supervision=True
        )

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        feat_maps = self.backbone(x)
        pred_mask = self.decoder(feat_maps)
        return pred_mask

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=CFG.lr)
        return [optimizer]


# ======================== Inference ========================

def predict_fn(test_loader, model, device, test_xyxys, pred_shape):
    mask_pred = np.zeros(pred_shape, dtype=np.float32)
    mask_count = np.zeros(pred_shape, dtype=np.float32)
    kernel = gkern(CFG.size, 1)
    kernel = kernel / kernel.max()
    model.eval()

    for step, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader)):
        images = images.to(device)
        with torch.no_grad():
            with torch.autocast(device_type="cuda"):
                y_preds = model(images)
        y_preds = torch.sigmoid(y_preds).to('cpu')
        for i, (x1, y1, x2, y2) in enumerate(xys):
            mask_pred[y1:y2, x1:x2] += np.multiply(
                F.interpolate(y_preds[i].unsqueeze(0).float(), scale_factor=4, mode='bilinear').squeeze(0).squeeze(0).numpy(),
                kernel)
            mask_count[y1:y2, x1:x2] += np.ones((CFG.size, CFG.size))

    mask_pred /= mask_count
    return mask_pred


from PIL import Image

CHECKPOINTS = [
    ('v2_ep13', './outputs/vesuvius/pretraining_all_3d_decoder_v2/vesuvius-models/r152_3ddec_v2_l5_0_crop_0_fr_i3depoch=13.ckpt'),
]

# (fragment_id, reverse_layers, base_path) tuples
segments_to_infer = [
    ('auto_grown_20251124-w020', True, '2um_dataset/1451_2um'),
    ('l_1', False, '2um_dataset/s4_2um'),
    ('l_2', False, '2um_dataset/s4_2um'),
]

for ckpt_tag, ckpt_path in CHECKPOINTS:
    print(f'\n{"="*60}', flush=True)
    print(f'Loading checkpoint: {ckpt_tag} ({ckpt_path})', flush=True)
    model = RegressionPLModel(pred_shape=(1, 1), enc='r152', with_norm=True)
    w = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(w['state_dict'])
    del w
    print('Checkpoint loaded', flush=True)

    model.cuda()
    model.eval()
    wandb.init(
        project="ink-experiments",
        entity='vesuvius-challenge',
        name=f"inference_3d_decoder_{ckpt_tag}",
    )

    for fragment_id, rev, base_path in segments_to_infer:
        rev_tag = 'rev' if rev else 'fwd'
        print(f'\n=== Inference: {fragment_id} ({rev_tag}, base={base_path}) [{ckpt_tag}] ===', flush=True)
        memmap_path = None
        try:
            test_loader, test_xyxz, test_shape, fragment_mask, memmap_path = get_img_splits(
                fragment_id, reverse_layers=rev, base_path=base_path)

            mask_pred = predict_fn(test_loader, model, device, test_xyxz, test_shape)
            mask_pred = np.clip(np.nan_to_num(mask_pred), a_min=0, a_max=1)
            mask_pred /= max(mask_pred.max(), 1e-8)

            # Save as PNG
            out_path = f'{fragment_id}_3ddec_{ckpt_tag}_pred.png'
            Image.fromarray((mask_pred * 255).astype(np.uint8)).save(out_path)
            print(f'  Saved {out_path} ({mask_pred.shape})', flush=True)

            # Log to wandb
            img = wandb.Image(mask_pred, caption=f"{fragment_id}_{rev_tag}_{ckpt_tag}")
            wandb.log({f'predictions/{fragment_id}_{rev_tag}': img})

            del test_loader, test_xyxz, fragment_mask, mask_pred
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as e:
            print(f'  ERROR on {fragment_id}: {e}', flush=True)
        finally:
            # Clean up memmap temp file if used
            if memmap_path and os.path.exists(memmap_path):
                os.unlink(memmap_path)
        print(f'  Finished {fragment_id}', flush=True)

    del model
    torch.cuda.empty_cache()
    gc.collect()
    wandb.finish()
    print(f'\nDone with {ckpt_tag}.', flush=True)

print('\nAll inference done.', flush=True)
