"""
Optimized inference module for ink detection using TimeSformer model.
Production-ready inference functions for processing scroll layers.
"""
import os
import gc
import logging
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from timesformer_pytorch import TimeSformer
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import cv2
from math import ceil, floor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

def gkern(h: int, w: int, sigma: float) -> np.ndarray:
    """
    Fast 2D Gaussian weight map of shape (h, w).
    """
    y = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
    x = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    xx, yy = np.meshgrid(x, y, indexing="xy")
    s2 = 2.0 * (sigma ** 2 if sigma > 0 else 1.0)
    k = np.exp(-(xx*xx + yy*yy) / s2)
    k_sum = k.sum()
    return k / (k_sum if k_sum > 0 else 1.0)

class InferenceConfig:
    """Configuration class for inference parameters"""
    # Model configuration
    in_chans = 26
    encoder_depth = 5
    
    # Inference configuration
    size = 64 # network input size (after resize)
    tile_size = 64
    stride = 32
    batch_size = 64
    workers = min(4, os.cpu_count() or 4)
    
    # Image processing
    max_clip_value = 200
    pad_size = 256
    
    # Prediction smoothing
    gaussian_sigma = 1

# Global config instance
CFG = InferenceConfig()

def preprocess_layers(layers: np.ndarray,
                     fragment_mask: Optional[np.ndarray] = None,
                     is_reverse_segment: bool = False) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:

    """
    Preprocess input layers for inference.
    
    Args:
        layers: numpy array of shape (H, W, C) where C is number of layers
        fragment_mask: Optional mask array of shape (H, W). If None, creates a white mask
        is_reverse_segment: Whether to reverse the layer order
        
    Returns:
        Tuple of (processed_layers, processed_mask)
    """
    try:
        # Validate input
        if layers.ndim != 3:
            raise ValueError(f"Expected 3D array (H, W, C), got shape {layers.shape}")
        
        if layers.shape[2] != CFG.in_chans:
            logger.warning(f"Expected {CFG.in_chans} channels, got {layers.shape[2]}")
        
        # Pad to ensure divisible by pad_size
        h, w, c = layers.shape
        orig_shape = (h, w)
        pad0 = (CFG.pad_size - h % CFG.pad_size) % CFG.pad_size
        pad1 = (CFG.pad_size - w % CFG.pad_size) % CFG.pad_size
        
        # Apply padding
        layers = np.pad(layers, [(0, pad0), (0, pad1), (0, 0)], constant_values=0)
        
        # Clip & cast to float32 early
        layers = np.clip(layers, 0, CFG.max_clip_value).astype(np.float32, copy=False)
        
        # Reverse if needed
        if is_reverse_segment:
            logger.info("Reversing segment layers")
            layers = layers[:, :, ::-1]
        
        # Process mask
        if fragment_mask is None:
            fragment_mask = np.ones((h, w), dtype=np.uint8) * 255
        
        # Pad mask to match layers
        fragment_mask = np.pad(fragment_mask, [(0, pad0), (0, pad1)], constant_values=0)
        
        logger.info(f"Preprocessed layers shape: {layers.shape}, mask shape: {fragment_mask.shape}")
        return layers, fragment_mask, orig_shape
        
    except Exception as e:
        logger.error(f"Error in preprocess_layers: {e}")
        raise

def create_inference_dataloader(layers: np.ndarray,
                               fragment_mask: np.ndarray) -> Tuple[DataLoader, Tuple[int, int]]:
    """
    Create a DataLoader for inference from preprocessed layers.
    
    Args:
        layers: Preprocessed layer array of shape (H, W, C)
        fragment_mask: Mask array of shape (H, W)
        
    Returns:
        Tuple of (dataloader, coordinates, original_shape)
    """
    try:     
        h, w, c = layers.shape
        
        # Generate sliding window coordinates
        x1_list = list(range(0, w - CFG.tile_size + 1, CFG.stride))
        y1_list = list(range(0, h - CFG.tile_size + 1, CFG.stride))
        
        # NOTE: keep only coordinates; do NOT precreate image tiles
        xyxys = []
        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                # Include tiles with ANY overlap with valid mask
                if np.any(fragment_mask[y1:y2, x1:x2] != 0):
                    xyxys.append([x1, y1, x2, y2])
        if not xyxys:
            raise ValueError("No valid tiles found in the input layers (mask fully empty?).")
        
        # Create dataset with transforms
        transform = A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.0] * CFG.in_chans,
                std=[1.0] * CFG.in_chans,
                max_pixel_value=CFG.max_clip_value
            ),
            ToTensorV2(),
        ])

        dataset = SlidingWindowDataset(layers, np.asarray(xyxys, dtype=np.int32), transform=transform)

        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        
        logger.info(f"Created dataloader with {len(dataset)} tiles")
        return dataloader, (h, w)
        
    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise

class SlidingWindowDataset(Dataset):
    """Tile dataset that materializes each tile lazily from the full layer array."""
    def __init__(self, layers: np.ndarray, xyxys: np.ndarray, transform=None):
        self.layers = layers  # (H, W, C)
        self.xyxys = xyxys    # (N, 4) int32
        self.transform = transform

    def __len__(self):
        return int(self.xyxys.shape[0])
    
    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.xyxys[idx].tolist()
        tile = self.layers[y1:y2, x1:x2, :]  # view; not pre-stored
        if self.transform:
            data = self.transform(image=tile)
            tile = data['image'].unsqueeze(0)  # (1, C, H, W) -> keep model input shape
        return tile, self.xyxys[idx]
    
class RegressionPLModel(pl.LightningModule):
    """TimeSformer model for ink detection inference"""
    
    def __init__(self, pred_shape=(1, 1), size=64, enc='', with_norm=False):
        super(RegressionPLModel, self).__init__()
        self.save_hyperparameters()
        
        # Model architecture
        self.backbone = TimeSformer(
            dim=512,
            image_size=64,
            patch_size=16,
            num_frames=CFG.in_chans,  # must match channel-depth we feed as frames
            num_classes=16,
            channels=1,
            depth=8,
            heads=6,
            dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1
        )
        
        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x):
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        x = self.backbone(torch.permute(x, (0, 2, 1, 3, 4)))
        x = x.view(-1, 1, 4, 4)
        return x


def predict_fn(test_loader: DataLoader, 
               model: RegressionPLModel, 
               device: torch.device, 
               pred_shape: Tuple[int, int]) -> np.ndarray:
    """
    Run inference on test data and return prediction mask.
    
    Args:
        test_loader: DataLoader with test data
        model: Trained model
        device: Torch device (cuda/cpu)
        test_xyxys: Array of tile coordinates
        pred_shape: Shape of output prediction
        
    Returns:
        Prediction mask as numpy array
    """
    try:
        mask_pred = np.zeros(pred_shape, dtype=np.float32)
        mask_count = np.zeros(pred_shape, dtype=np.float32)

        # weights will match the resized prediction spatial size (set below)
        kernel_tensor = None
        
        model.eval()
        
        with torch.no_grad():
            for step, (images, xys) in tqdm(enumerate(test_loader), desc="Running inference"):
                images = images.to(device)
                
                # Forward pass with autocast for efficiency
                amp_device = "cuda" if device.type == "cuda" else "cpu"
                with torch.autocast(device_type=amp_device):
                    y_preds = model(images)
                
                # Apply sigmoid and resize predictions
                y_preds = torch.sigmoid(y_preds)
                # ALWAYS resize to tile_size so placement = (y1:y2, x1:x2)
                y_preds_resized = F.interpolate(
                    y_preds.float(),
                    size=(CFG.tile_size, CFG.tile_size),
                    mode='bilinear',
                    align_corners=False
                )  # (B,1,tile,tile)
                # Build Gaussian weights once with the correct size
                if kernel_tensor is None:
                    kh, kw = y_preds_resized.shape[-2], y_preds_resized.shape[-1]
                    kernel_np = gkern(kh, kw, CFG.gaussian_sigma).astype(np.float32)
                    kernel_tensor = torch.from_numpy(kernel_np).to(device)  # (kh,kw)
                
                # Weight predictions
                y_preds_weighted = (y_preds_resized * kernel_tensor).squeeze(1)  # (B, H, W)
                
                # Move to CPU for accumulation
                y_preds_cpu = y_preds_weighted.cpu().numpy()
                weight_cpu = kernel_tensor.detach().cpu().numpy().astype(np.float32)
                
                # Accumulate predictions
                if torch.is_tensor(xys):
                    xys = xys.cpu().numpy().astype(np.int32)
                for i in range(xys.shape[0]):
                    x1, y1, x2, y2 = [int(v) for v in xys[i]]
                    mask_pred[y1:y2, x1:x2] += y_preds_cpu[i]
                    mask_count[y1:y2, x1:x2] += weight_cpu
        
        # Normalize by count to handle overlapping regions
        mask_pred = mask_pred / np.clip(mask_count, a_min=1e-6, a_max=None)
        
        # Clip and normalize to [0, 1]
        mask_pred = np.clip(mask_pred, 0, 1)
        
        logger.info(f"Inference completed. Prediction shape: {mask_pred.shape}")
        return mask_pred
        
    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        raise

def run_inference(layers: np.ndarray,
                  model: RegressionPLModel,
                  device: torch.device,
                  fragment_mask: Optional[np.ndarray] = None,
                  is_reverse_segment: bool = False) -> np.ndarray:
    """
    Main inference function that processes layer data and returns prediction mask.
    
    Args:
        layers: Input layer data as numpy array (H, W, C)
        model: Loaded TimeSformer model
        device: Torch device
        fragment_mask: Optional mask for valid regions
        is_reverse_segment: Whether to reverse layer order
        
    Returns:
        Prediction mask as numpy array with values in [0, 1]
    """
    try:
        logger.info("Starting inference process...")
        
        # Preprocess layers
        processed_layers, processed_mask, orig_shape = preprocess_layers(
            layers, fragment_mask, is_reverse_segment
        )
        
        # Create dataloader
        test_loader, test_xyxys, pred_shape = create_inference_dataloader(
            processed_layers, processed_mask
        )
        
        # Run inference
        mask_pred = predict_fn(test_loader, model, device, pred_shape)
        
        # Post-process results
        mask_pred = np.clip(np.nan_to_num(mask_pred), 0, 1)
        
        # Crop back to the original unpadded size
        oh, ow = orig_shape
        mask_pred = mask_pred[:oh, :ow]
        # Optional min-max normalization on the valid area only
        mx = mask_pred.max()
        if mx > 0:
            mask_pred = mask_pred / mx
        
        logger.info("Inference completed successfully")
        return mask_pred
        
    except Exception as e:
        logger.error(f"Error in run_inference: {e}")
        raise
    finally:
        # Cleanup
        try: del test_loader
        except: pass
        torch.cuda.empty_cache()
        gc.collect()
