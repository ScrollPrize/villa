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
import scipy.stats as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

class InferenceConfig:
    """Configuration class for inference parameters"""
    # Model configuration
    in_chans = 26
    encoder_depth = 5
    
    # Inference configuration
    size = 64
    tile_size = 64
    stride = 32
    batch_size = 64
    workers = 4
    
    # Image processing
    max_clip_value = 200
    pad_size = 256
    
    # Prediction smoothing
    gaussian_kernel_size = 64
    gaussian_sigma = 1

# Global config instance
CFG = InferenceConfig()

def preprocess_layers(layers: np.ndarray, 
                     fragment_mask: Optional[np.ndarray] = None,
                     is_reverse_segment: bool = False) -> Tuple[np.ndarray, np.ndarray]:
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
        pad0 = (CFG.pad_size - h % CFG.pad_size) % CFG.pad_size
        pad1 = (CFG.pad_size - w % CFG.pad_size) % CFG.pad_size
        
        # Apply padding
        layers = np.pad(layers, [(0, pad0), (0, pad1), (0, 0)], constant_values=0)
        
        # Clip values
        layers = np.clip(layers, 0, CFG.max_clip_value)
        
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
        return layers, fragment_mask
        
    except Exception as e:
        logger.error(f"Error in preprocess_layers: {e}")
        raise

def create_inference_dataloader(layers: np.ndarray, 
                               fragment_mask: np.ndarray) -> Tuple[DataLoader, np.ndarray, Tuple[int, int]]:
    """
    Create a DataLoader for inference from preprocessed layers.
    
    Args:
        layers: Preprocessed layer array of shape (H, W, C)
        fragment_mask: Mask array of shape (H, W)
        
    Returns:
        Tuple of (dataloader, coordinates, original_shape)
    """
    try:
        images = []
        xyxys = []
        
        h, w, c = layers.shape
        
        # Generate sliding window coordinates
        x1_list = list(range(0, w - CFG.tile_size + 1, CFG.stride))
        y1_list = list(range(0, h - CFG.tile_size + 1, CFG.stride))
        
        for y1 in y1_list:
            for x1 in x1_list:
                y2 = y1 + CFG.tile_size
                x2 = x1 + CFG.tile_size
                
                # Only include tiles where mask is not zero (valid regions)
                if not np.any(fragment_mask[y1:y2, x1:x2] == 0):
                    tile = layers[y1:y2, x1:x2]
                    images.append(tile)
                    xyxys.append([x1, y1, x2, y2])
        
        if not images:
            raise ValueError("No valid tiles found in the input layers")
        
        # Create dataset with transforms
        transform = A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0] * CFG.in_chans,
                std=[1] * CFG.in_chans
            ),
            ToTensorV2(transpose_mask=True),
        ])
        
        dataset = CustomDatasetTest(images, np.stack(xyxys), transform=transform)
        
        # Create DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.workers,
            pin_memory=True,
            drop_last=False,
        )
        
        logger.info(f"Created dataloader with {len(images)} tiles")
        return dataloader, np.stack(xyxys), (h, w)
        
    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise

class CustomDatasetTest(Dataset):
    """Dataset class for inference on image tiles"""
    
    def __init__(self, images: List[np.ndarray], xyxys: np.ndarray, transform=None):
        self.images = images
        self.xyxys = xyxys
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        xy = self.xyxys[idx]
        
        if self.transform:
            data = self.transform(image=image)
            image = data['image'].unsqueeze(0)
        
        return image, xy
    
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
            num_frames=30,
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
               test_xyxys: np.ndarray, 
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
        mask_count_kernel = np.ones((CFG.size, CFG.size), dtype=np.float32)
        
        # Create Gaussian kernel for smoothing
        kernel = gkern(CFG.gaussian_kernel_size, CFG.gaussian_sigma)
        kernel = kernel / kernel.max()
        kernel_tensor = torch.tensor(kernel, device=device, dtype=torch.float32)
        
        model.eval()
        
        with torch.no_grad():
            for step, (images, xys) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Running inference"):
                images = images.to(device)
                
                # Forward pass with autocast for efficiency
                with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    y_preds = model(images)
                
                # Apply sigmoid and resize predictions
                y_preds = torch.sigmoid(y_preds)
                y_preds_resized = F.interpolate(
                    y_preds.float(), 
                    scale_factor=16, 
                    mode='bilinear', 
                    align_corners=False
                )
                
                # Apply Gaussian smoothing
                y_preds_smoothed = y_preds_resized * kernel_tensor
                y_preds_smoothed = y_preds_smoothed.squeeze(1)
                
                # Move to CPU for accumulation
                y_preds_cpu = y_preds_smoothed.cpu().numpy()
                
                # Accumulate predictions
                for i, (x1, y1, x2, y2) in enumerate(xys):
                    mask_pred[y1:y2, x1:x2] += y_preds_cpu[i]
                    mask_count[y1:y2, x1:x2] += mask_count_kernel
        
        # Normalize by count to handle overlapping regions
        mask_pred = mask_pred / np.clip(mask_count, a_min=1, a_max=None)
        
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
        processed_layers, processed_mask = preprocess_layers(
            layers, fragment_mask, is_reverse_segment
        )
        
        # Create dataloader
        test_loader, test_xyxys, pred_shape = create_inference_dataloader(
            processed_layers, processed_mask
        )
        
        # Run inference
        mask_pred = predict_fn(test_loader, model, device, test_xyxys, pred_shape)
        
        # Post-process results
        mask_pred = np.clip(np.nan_to_num(mask_pred), 0, 1)
        
        # Normalize to [0, 1] if there are any predictions
        if mask_pred.max() > 0:
            mask_pred = mask_pred / mask_pred.max()
        
        logger.info("Inference completed successfully")
        return mask_pred
        
    except Exception as e:
        logger.error(f"Error in run_inference: {e}")
        raise
    finally:
        # Cleanup
        try:
            del test_loader
        except:
            pass
        torch.cuda.empty_cache()
        gc.collect()
