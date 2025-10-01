"""
Optimized inference module for ink detection using TimeSformer model.
Production-ready, memory-friendly inference for processing scroll layers.

Key features:
- No full HxWxC tensor in RAM (stream tiles from disk or accept a legacy numpy array)
- Correct TimeSformer framing (frames=C, channels=1)
- Smooth overlap-add with a Hann window (no dotted grid)
- Efficient dataloader settings and safe zero-padding at image borders
"""
import os
import gc
import uuid
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

# ----------------------------- Logging ---------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

# ----------------------------- Helpers ---------------------------------------
def gkern(h: int, w: int, sigma: float) -> np.ndarray:
    """Normalized 2D Gaussian (sum=1)."""
    y = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
    x = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    xx, yy = np.meshgrid(x, y, indexing="xy")
    s2 = 2.0 * (sigma ** 2 if sigma > 0 else 1.0)
    k = np.exp(-(xx * xx + yy * yy) / s2)
    s = k.sum()
    return k / (s if s > 0 else 1.0)

def hann2d(h: int, w: int) -> np.ndarray:
    """Normalized 2D Hann window (sum=1) for overlap-add blending."""
    wy = np.hanning(h).astype(np.float32)
    wx = np.hanning(w).astype(np.float32)
    k = np.outer(wy, wx)
    s = k.sum()
    return k / (s if s > 0 else 1.0)

def _grid_1d(L: int, tile: int, stride: int) -> List[int]:
    """1D positions covering [0..L) and forcing the last tile to touch the border."""
    xs = list(range(0, max(1, L - tile + 1), stride))
    end = max(0, L - tile)
    if not xs or xs[-1] != end:
        xs.append(end)
    return xs

# ----------------------------- Config ----------------------------------------
class InferenceConfig:
    # Model configuration
    in_chans = 26  # frames
    encoder_depth = 5

    # Inference configuration
    size = 64       # net input size (post-resize, normally equals tile_size)
    tile_size = 64
    stride = 16
    batch_size = 64
    workers = min(4, os.cpu_count() or 4)

    # Image processing / scaling
    max_clip_value = 200

    # Blending
    use_hann_window = True
    gaussian_sigma = 0.0  # if using Gaussian: 0 -> auto (~ tile/2.5)

    # Tile selection
    min_valid_ratio = 0.0  # keep tiles with >= this fraction of nonzero mask

CFG = InferenceConfig()

# --------------------- Disk-backed / Array-backed layers ---------------------
class LayersSource:
    """
    Unified source of layers that supports:
      • numpy arrays of shape (H, W, C)
      • list of file paths to grayscale images (C files)

    When given file paths, builds a disk-backed memmap (uint8) (H, W, C) once,
    so tiles are read quickly without holding the whole stack in RAM.
    """
    def __init__(self, src: Union[np.ndarray, List[str]]):
        if isinstance(src, np.ndarray):
            if src.ndim != 3:
                raise ValueError(f"Expected (H,W,C) array, got {src.shape}")
            self._arr = src
            self._mm = None
            self._shape = src.shape
            self._dtype = src.dtype
            self._owns_mm = False
        elif isinstance(src, list):
            if len(src) != CFG.in_chans:
                logger.warning(f"Expected {CFG.in_chans} layer files, got {len(src)}")
            self._arr = None
            self._mm, self._shape, self._dtype = self._build_memmap_from_files(src)
            self._owns_mm = True
        else:
            raise TypeError("LayersSource expects np.ndarray or List[str] of file paths")

    @staticmethod
    def _build_memmap_from_files(paths: List[str]) -> Tuple[np.memmap, Tuple[int, int, int], np.dtype]:
        # Probe first image
        first = cv2.imread(paths[0], cv2.IMREAD_GRAYSCALE)
        if first is None:
            raise RuntimeError(f"Failed to read image: {paths[0]}")
        h, w = first.shape
        c = len(paths)

        # Create a memmap file on disk (uint8 keeps size modest)
        mm_path = os.path.join("/tmp", f"layers_{uuid.uuid4().hex}.npy")
        logger.info(f"Creating disk-backed memmap for layers at {mm_path} (shape={h}x{w}x{c}, dtype=uint8)")
        mm = np.memmap(mm_path, mode="w+", dtype=np.uint8, shape=(h, w, c))

        # Stream images one-by-one, validate shape, write to memmap
        for idx, p in enumerate(paths):
            img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise RuntimeError(f"Failed to read image: {p}")
            if img.shape != (h, w):
                raise RuntimeError(f"Layer size mismatch: {p} has {img.shape}, expected {(h, w)}")
            # No global clipping; we clip per-tile before normalize
            mm[:, :, idx] = img

        # Flush to disk, reopen read-only for safety/perf
        del mm
        mm = np.memmap(mm_path, mode="r", dtype=np.uint8, shape=(h, w, c))
        return mm, (h, w, c), mm.dtype

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    def read_roi(self, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
        """Read ROI with zero-padding for out-of-bounds, returns (tile_h, tile_w, C)."""
        H, W, C = self._shape
        yy1, yy2 = max(0, y1), min(H, y2)
        xx1, xx2 = max(0, x1), min(W, x2)
        out = np.zeros((y2 - y1, x2 - x1, C),
                       dtype=self._dtype if self._arr is None else self._arr.dtype)
        if yy2 > yy1 and xx2 > xx1:
            if self._arr is not None:
                out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = self._arr[yy1:yy2, xx1:xx2, :]
            else:
                out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = self._mm[yy1:yy2, xx1:xx2, :]
        return out

# ----------------------------- Preprocess ------------------------------------
def preprocess_layers(
    layers: Union[np.ndarray, List[str]],
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False
) -> Tuple[LayersSource, np.ndarray, Tuple[int, int], bool]:
    """
    Prepare layers for streaming inference.

    Returns:
        (source, mask, orig_shape, reverse_flag)
    """
    try:
        src = LayersSource(layers)  # could build memmap here
        h, w, c = src.shape
        if c != CFG.in_chans:
            logger.warning(f"Model expects {CFG.in_chans} channels, got {c}")

        if fragment_mask is None:
            fragment_mask = np.ones((h, w), dtype=np.uint8) * 255

        logger.info(f"Prepared layers source: shape={src.shape}, mask={fragment_mask.shape}, reverse={is_reverse_segment}")
        return src, fragment_mask, (h, w), bool(is_reverse_segment)

    except Exception as e:
        logger.error(f"Error in preprocess_layers: {e}")
        raise

# ---------------------------- Dataloader -------------------------------------
class SlidingWindowDataset(Dataset):
    """
    Lazily materializes (tile_h, tile_w, C) from the source object per tile,
    applies clipping and transforms, and returns tensors as (1, C, H, W).
    """
    def __init__(self, source: LayersSource, xyxys: np.ndarray, reverse: bool, transform):
        self.source = source
        self.xyxys = xyxys.astype(np.int32, copy=False)
        self.reverse = reverse
        self.transform = transform

    def __len__(self) -> int:
        return int(self.xyxys.shape[0])

    def __getitem__(self, idx):
        x1, y1, x2, y2 = self.xyxys[idx].tolist()
        tile = self.source.read_roi(y1, y2, x1, x2)  # (tile, tile, C), uint8
        if self.reverse:
            tile = tile[:, :, ::-1]
        # Clip to match training range
        tile = np.clip(tile, 0, CFG.max_clip_value).astype(tile.dtype, copy=False)

        data = self.transform(image=tile)  # -> tensor (C,H,W)
        tens = data["image"].unsqueeze(0)  # -> (1,C,H,W) so C becomes frames
        return tens, self.xyxys[idx]

def create_inference_dataloader(
    source: LayersSource,
    fragment_mask: np.ndarray,
    reverse: bool
) -> Tuple[DataLoader, Tuple[int, int]]:
    """Return (loader, pred_shape=(H,W))."""
    try:
        h, w, _ = source.shape

        x1_list = _grid_1d(w, CFG.tile_size, CFG.stride)
        y1_list = _grid_1d(h, CFG.tile_size, CFG.stride)

        xyxys: List[List[int]] = []
        for y1 in y1_list:
            for x1 in x1_list:
                y2, x2 = y1 + CFG.tile_size, x1 + CFG.tile_size
                # compute valid ratio inside bounds
                yy1, yy2 = max(0, y1), min(h, y2)
                xx1, xx2 = max(0, x1), min(w, x2)
                roi = fragment_mask[yy1:yy2, xx1:xx2]
                valid_ratio = float(roi.size and (roi != 0).mean() or 0.0)
                if valid_ratio >= CFG.min_valid_ratio:
                    xyxys.append([x1, y1, x2, y2])

        if not xyxys:
            raise ValueError("No valid tiles (mask empty or fully filtered).")

        # Build transforms; avoid no-op resize
        tfm_list = []
        if CFG.tile_size != CFG.size:
            tfm_list.append(A.Resize(CFG.size, CFG.size))
        tfm_list += [
            A.Normalize(mean=[0.0] * CFG.in_chans, std=[1.0] * CFG.in_chans,
                        max_pixel_value=CFG.max_clip_value),
            ToTensorV2(),
        ]
        transform = A.Compose(tfm_list)

        dataset = SlidingWindowDataset(source, np.asarray(xyxys), reverse, transform)

        loader = DataLoader(
            dataset,
            batch_size=CFG.batch_size,
            shuffle=False,
            num_workers=CFG.workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(CFG.workers > 0),
            prefetch_factor=2 if CFG.workers > 0 else None,
            drop_last=False,
        )
        logger.info(f"Created dataloader with {len(dataset)} tiles")
        return loader, (h, w)

    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise

# ------------------------------- Model ---------------------------------------
class RegressionPLModel(pl.LightningModule):
    """TimeSformer for ink detection inference."""
    def __init__(self, pred_shape=(1, 1), size=64, enc='', with_norm=False):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = TimeSformer(
            dim=512,
            image_size=64,
            patch_size=16,
            num_frames=CFG.in_chans,   # frames = layers
            num_classes=16,            # 4x4 logits
            channels=1,                # single-channel per frame
            depth=8,
            heads=6,
            dim_head=64,
            attn_dropout=0.1,
            ff_dropout=0.1,
        )

        if self.hparams.with_norm:
            self.normalization = nn.BatchNorm3d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input from dataset: (B,1,C,H,W). TimeSformer lib expects (B,frames,channels,H,W).
        if x.ndim == 4:
            x = x[:, None]
        if self.hparams.with_norm:
            x = self.normalization(x)
        x = torch.permute(x, (0, 2, 1, 3, 4))  # -> (B,C,1,H,W)
        x = self.backbone(x)
        x = x.view(-1, 1, 4, 4)                # (B,1,4,4)
        return x

# ----------------------------- Inference -------------------------------------
def predict_fn(
    test_loader: DataLoader,
    model: RegressionPLModel,
    device: torch.device,
    pred_shape: Tuple[int, int]
) -> np.ndarray:
    """Run tiled inference and blend into a single (H,W) heatmap."""
    try:
        H, W = pred_shape
        mask_pred = np.zeros((H, W), dtype=np.float32)
        mask_count = np.zeros((H, W), dtype=np.float32)

        weight_tensor: Optional[torch.Tensor] = None
        model.eval()

        with torch.inference_mode():
            # total tiles = len(dataset); progress bar advances by batch size
            try:
                total_tiles = len(test_loader.dataset)
            except Exception:
                total_tiles = None
            pbar = tqdm(total=total_tiles,
                        desc="Running inference",
                        unit="tile",
                        dynamic_ncols=True)

            for (images, xys) in test_loader:
                images = images.to(device, non_blocking=True)

                amp_device = "cuda" if device.type == "cuda" else "cpu"
                with torch.autocast(device_type=amp_device, enabled=True):
                    y_preds = model(images)  # (B,1,4,4)

                y_preds = torch.sigmoid(y_preds)
                y_preds_resized = F.interpolate(
                    y_preds.float(),
                    size=(CFG.tile_size, CFG.tile_size),
                    mode='bilinear',
                    align_corners=False
                )  # (B,1,tile,tile)

                if weight_tensor is None:
                    th, tw = y_preds_resized.shape[-2:]
                    if CFG.use_hann_window:
                        w_np = hann2d(th, tw).astype(np.float32)
                    else:
                        sigma = CFG.gaussian_sigma if CFG.gaussian_sigma > 0 else max(1.0, min(th, tw) / 2.5)
                        w_np = gkern(th, tw, sigma).astype(np.float32)
                    weight_tensor = torch.from_numpy(w_np).to(device)  # (th,tw)

                y_weighted = (y_preds_resized * weight_tensor).squeeze(1)  # (B,th,tw)

                y_cpu = y_weighted.cpu().numpy()
                w_cpu = weight_tensor.detach().cpu().numpy().astype(np.float32)

                if torch.is_tensor(xys):
                    xys = xys.cpu().numpy().astype(np.int32)
                for i in range(xys.shape[0]):
                    x1, y1, x2, y2 = [int(v) for v in xys[i]]
                    mask_pred[y1:y2, x1:x2] += y_cpu[i]
                    mask_count[y1:y2, x1:x2] += w_cpu
                # advance by the number of tiles in this batch
                pbar.update(images.size(0))
            pbar.close()

        mask_pred = mask_pred / np.clip(mask_count, a_min=1e-6, a_max=None)
        mask_pred = np.clip(mask_pred, 0, 1)
        logger.info(f"Inference completed. Prediction shape: {mask_pred.shape}")
        return mask_pred

    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        raise

def run_inference(
    layers: Union[np.ndarray, List[str]],
    model: RegressionPLModel,
    device: torch.device,
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False
) -> np.ndarray:
    """
    Main entrypoint: accepts either a stacked array (H,W,C) or a list of file paths.
    Returns (H,W) heatmap cropped to the original size.
    """
    try:
        logger.info("Starting inference process...")
        source, mask, orig_shape, reverse = preprocess_layers(
            layers, fragment_mask, is_reverse_segment
        )
        test_loader, pred_shape = create_inference_dataloader(source, mask, reverse)
        mask_pred = predict_fn(test_loader, model, device, pred_shape)

        # Crop to original size and re-normalize per-fragment
        oh, ow = orig_shape
        mask_pred = np.clip(mask_pred[:oh, :ow], 0, 1)
        mx = float(mask_pred.max())
        if mx > 0:
            mask_pred = mask_pred / mx

        logger.info("Inference completed successfully")
        return mask_pred

    except Exception as e:
        logger.error(f"Error in run_inference: {e}")
        raise
    finally:
        try:
            del test_loader
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
