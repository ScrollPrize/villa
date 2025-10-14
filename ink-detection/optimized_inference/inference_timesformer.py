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
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "0")  # safety in case imported directly
import gc
import uuid
import logging
import shutil
import math
from typing import List, Tuple, Optional, Union, Dict

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
import tifffile as tiff
import cv2
import zarr

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
    batch_size = 256  # Increased for better GPU utilization
    workers = min(8, os.cpu_count() or 4)  # More workers to feed GPU faster
    prefetch_factor = 4  # More prefetching per worker

    # Image processing / scaling
    max_clip_value = 200

    # Blending
    use_hann_window = True
    gaussian_sigma = 0.0  # if using Gaussian: 0 -> auto (~ tile/2.5)

    # Tile selection
    min_valid_ratio = 0.0  # keep tiles with >= this fraction of nonzero mask

    # Partitioning for map/reduce inference
    num_parts = 1  # Total number of partitions (1 = no partitioning)
    part_id = 0    # Current partition ID (0-indexed)
    zarr_output_dir = os.environ.get("ZARR_OUTPUT_DIR", "/tmp/partitions")

CFG = InferenceConfig()

# --------------------- Surface Volume Zarr Creation -------------------------
def _read_gray_any(path: str) -> np.ndarray:
    """Read a grayscale image from various formats."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".tif", ".tiff"):
            img = tiff.imread(path)
            if img is None:
                logger.warning(f"Failed to read image: {path}")
                return None
            if img.ndim > 2:
                img = img[..., 0]
            if img.dtype != np.uint8:
                # Minimal, safe conversion to uint8
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img
        else:
            return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    except Exception:
        logger.exception(f"Exception reading image: {path}")
        return None

def create_surface_volume_zarr(
    layer_paths: List[str],
    output_path: str,
    chunk_size: int = 1024,
    max_workers: Optional[int] = None
) -> str:
    """
    Create a surface volume zarr array from a list of layer image files.

    Args:
        layer_paths: List of paths to layer image files (must be sorted in layer order)
        output_path: Path where the zarr array will be created
        chunk_size: Chunk size for zarr array (default: 1024x1024x1)
        max_workers: Number of worker threads for parallel reading (default: auto)

    Returns:
        Path to the created zarr array (same as output_path)

    Raises:
        RuntimeError: If image reading fails or size mismatches occur
    """
    import concurrent.futures
    from numcodecs import LZ4

    if not layer_paths:
        raise ValueError("layer_paths cannot be empty")

    # Read first image to get dimensions
    first = _read_gray_any(layer_paths[0])
    if first is None:
        raise RuntimeError(f"Failed to read image: {layer_paths[0]}")
    h, w = first.shape
    c = len(layer_paths)

    # Check disk space (estimate with 50% compression ratio for LZ4)
    zarr_root = os.path.dirname(output_path) or "/tmp"
    uncompressed_gib = (h * w * c) / (1024**3)
    estimated_gib = uncompressed_gib * 0.5  # LZ4 typically achieves ~50% compression on uint8 images
    free_gib = shutil.disk_usage(zarr_root).free / (1024**3)
    logger.info(
        f"Creating surface volume zarr at {output_path} (H,W,C={h},{w},{c} ~{uncompressed_gib:.2f} GiB uncompressed, "
        f"~{estimated_gib:.2f} GiB estimated with LZ4). Free on {zarr_root}: {free_gib:.2f} GiB."
    )
    if free_gib < estimated_gib * 1.10:
        raise RuntimeError(
            f"Not enough space on {zarr_root}: need ~{estimated_gib:.2f} GiB, have {free_gib:.2f} GiB"
        )

    # Remove existing zarr if present
    if os.path.exists(output_path):
        logger.warning(f"Removing existing zarr at {output_path}")
        shutil.rmtree(output_path)

    # Create zarr v2 array with shape (H, W, C) and chunk size optimized for spatial tile access
    # Using LZ4 compression (fast with decent ratio)
    z = zarr.open(
        output_path,
        mode="w",
        shape=(h, w, c),
        chunks=(chunk_size, chunk_size, 1),
        dtype=np.uint8,
        compressor=LZ4(acceleration=1),  # acceleration=1 is default, good balance of speed/ratio
        write_empty_chunks=False  # Don't write chunks that are entirely fill_value
    )

    # Parallel reads with batching to limit memory usage
    if max_workers is None:
        max_workers = int(os.environ.get("MMAP_READ_WORKERS", str(min(6, (os.cpu_count() or 4)))))
    log_every = max(1, c // 10)
    # Limit in-flight futures to prevent memory buildup
    batch_size = max_workers * 2

    def _read_and_write(idx_path):
        """Read image and write directly to zarr, return only index."""
        idx, p = idx_path
        img = _read_gray_any(p)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        if img.shape != (h, w):
            raise RuntimeError(f"Layer size mismatch: {p} has {img.shape}, expected {(h, w)}")
        z[:, :, idx] = img  # write to zarr immediately
        del img  # release memory ASAP
        return idx

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Process in batches to limit memory
        for batch_start in range(0, c, batch_size):
            batch_end = min(batch_start + batch_size, c)
            batch_paths = [(i, layer_paths[i]) for i in range(batch_start, batch_end)]
            futures = [ex.submit(_read_and_write, ip) for ip in batch_paths]

            for fut in concurrent.futures.as_completed(futures):
                idx = fut.result()
                n_done += 1
                if (n_done % log_every == 0) or (n_done == c):
                    logger.info(f"Surface volume zarr build progress: {n_done}/{c} layers")

    logger.info(f"Surface volume zarr created successfully at {output_path}")
    return output_path

# --------------------- Disk-backed / Array-backed layers ---------------------
class LayersSource:
    """
    Unified source of layers that supports:
      • numpy arrays of shape (H, W, C)
      • path to an existing zarr array

    For reading tiles during inference without holding the whole stack in RAM.
    """
    def __init__(self, src: Union[np.ndarray, str]):
        if isinstance(src, np.ndarray):
            if src.ndim != 3:
                raise ValueError(f"Expected (H,W,C) array, got {src.shape}")
            self._arr = src
            self._mm = None
            self._shape = src.shape
            self._dtype = src.dtype
        elif isinstance(src, str):
            # Path to existing zarr array
            if not os.path.exists(src):
                raise ValueError(f"Zarr path does not exist: {src}")
            logger.info(f"Opening existing zarr array at {src}")
            self._arr = None
            self._mm = zarr.open(src, mode='r')
            self._shape = self._mm.shape
            self._dtype = self._mm.dtype
            if len(self._shape) != 3:
                raise ValueError(f"Expected (H,W,C) zarr, got shape {self._shape}")
            logger.info(f"Loaded zarr with shape {self._shape}, dtype {self._dtype}")
        else:
            raise TypeError("LayersSource expects np.ndarray or str (zarr path)")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    def read_roi(self, y1: int, y2: int, x1: int, x2: int) -> np.ndarray:
        """Read ROI with zero-padding for out-of-bounds, returns (tile_h, tile_w, C)."""
        H, W, C = self._shape
        yy1, yy2 = max(0, y1), min(H, y2)
        xx1, xx2 = max(0, x1), min(W, x2)
        out = np.zeros(
            (y2 - y1, x2 - x1, C),
            dtype=self._dtype if self._arr is None else self._arr.dtype,
        )
        if yy2 > yy1 and xx2 > xx1:
            if self._arr is not None:
                # in-RAM case is already HWC
                out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = self._arr[yy1:yy2, xx1:xx2, :]
            else:
                # zarr array stored as (H,W,C) -> slice directly
                roi = self._mm[yy1:yy2, xx1:xx2, :]
                out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = roi
        return out

# ----------------------------- Preprocess ------------------------------------
def preprocess_layers(
    layers: Union[np.ndarray, str],
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False
) -> Tuple[LayersSource, np.ndarray, Tuple[int, int], bool]:
    """
    Prepare layers for streaming inference.

    Args:
        layers: Either a numpy array (H, W, C) or path to a zarr array
        fragment_mask: Optional mask array (H, W)
        is_reverse_segment: Whether to reverse layer order

    Returns:
        (source, mask, orig_shape, reverse_flag)
    """
    try:
        src = LayersSource(layers)
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
        # Clip to match training range - in-place for speed
        np.clip(tile, 0, CFG.max_clip_value, out=tile)

        data = self.transform(image=tile)  # -> tensor (C,H,W)
        tens = data["image"].unsqueeze(0)  # -> (1,C,H,W) so C becomes frames
        return tens, self.xyxys[idx]

def create_inference_dataloader(
    source: LayersSource,
    fragment_mask: np.ndarray,
    reverse: bool
) -> Tuple[DataLoader, Tuple[int, int], Dict[str, int]]:
    """Return (loader, pred_shape=(H,W), partition_info)."""
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

        total_tiles = len(xyxys)

        # Apply range-based partitioning if num_parts > 1
        partition_info = {
            "total_tiles": total_tiles,
            "start_idx": 0,
            "end_idx": total_tiles,
            "partition_tiles": total_tiles,
        }

        if CFG.num_parts > 1:
            tiles_per_part = math.ceil(total_tiles / CFG.num_parts)
            start_idx = CFG.part_id * tiles_per_part
            end_idx = min(start_idx + tiles_per_part, total_tiles)
            xyxys = xyxys[start_idx:end_idx]

            partition_info.update({
                "start_idx": start_idx,
                "end_idx": end_idx,
                "partition_tiles": len(xyxys),
            })

            logger.info(
                f"Partition {CFG.part_id}/{CFG.num_parts}: "
                f"processing tiles [{start_idx}:{end_idx}] "
                f"({len(xyxys)} tiles out of {total_tiles} total)"
            )

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
            prefetch_factor=CFG.prefetch_factor if CFG.workers > 0 else None,
            drop_last=False,
            multiprocessing_context='fork' if CFG.workers > 0 else None,  # Faster worker startup
        )
        logger.info(f"Created dataloader with {len(dataset)} tiles")
        return loader, (h, w), partition_info

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
) -> Union[np.ndarray, Dict[str, str]]:
    """
    Run tiled inference and blend into a single (H,W) heatmap.

    Returns:
        - If num_parts == 1: returns blended numpy array (H, W)
        - If num_parts > 1: returns dict with zarr paths {"mask_pred": path, "mask_count": path}
    """
    try:
        H, W = pred_shape

        # Always use in-memory numpy arrays during inference loop for speed
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

        # Write results based on mode
        if CFG.num_parts > 1:
            # Partitioned mode: write in-memory arrays to zarr once at the end
            from numcodecs import LZ4

            logger.info(f"Writing partition {CFG.part_id} results to zarr arrays...")
            os.makedirs(CFG.zarr_output_dir, exist_ok=True)

            mask_pred_path = os.path.join(CFG.zarr_output_dir, f"mask_pred_part_{CFG.part_id:03d}.zarr")
            mask_count_path = os.path.join(CFG.zarr_output_dir, f"mask_count_part_{CFG.part_id:03d}.zarr")

            # Create and write mask_pred zarr with LZ4 compression
            mask_pred_z = zarr.open(
                mask_pred_path,
                mode='w',
                shape=(H, W),
                chunks=(1024, 1024),
                dtype=np.float32,
                compressor=LZ4(acceleration=1),
                write_empty_chunks=False
            )
            mask_pred_z[:] = mask_pred

            # Create and write mask_count zarr with LZ4 compression
            mask_count_z = zarr.open(
                mask_count_path,
                mode='w',
                shape=(H, W),
                chunks=(1024, 1024),
                dtype=np.float32,
                compressor=LZ4(acceleration=1),
                write_empty_chunks=False
            )
            mask_count_z[:] = mask_count

            logger.info(f"Partition {CFG.part_id} completed. Wrote zarr arrays to {CFG.zarr_output_dir}")
            return {
                "mask_pred": mask_pred_path,
                "mask_count": mask_count_path,
            }

        # Standard mode: blend and return numpy array
        mask_pred = mask_pred / np.clip(mask_count, a_min=1e-6, a_max=None)
        mask_pred = np.clip(mask_pred, 0, 1)
        logger.info(f"Inference completed. Prediction shape: {mask_pred.shape}")
        return mask_pred

    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        raise

def reduce_partitions(
    zarr_output_dir: str,
    num_parts: int,
    pred_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Reduce/blend all partition zarr arrays into a single prediction.

    Args:
        zarr_output_dir: Directory containing partition zarr arrays
        num_parts: Total number of partitions
        pred_shape: Output shape (H, W)

    Returns:
        Blended numpy array (H, W) with values in [0, 1]
    """
    try:
        H, W = pred_shape
        logger.info(f"Starting reduce phase: blending {num_parts} partitions from {zarr_output_dir}")

        # Initialize accumulators in memory (we're blending, so need the full arrays)
        total_pred = np.zeros((H, W), dtype=np.float32)
        total_count = np.zeros((H, W), dtype=np.float32)

        # Process each partition in chunks to avoid memory issues
        chunk_size = 1024
        for part_id in range(num_parts):
            mask_pred_path = os.path.join(zarr_output_dir, f"mask_pred_part_{part_id:03d}.zarr")
            mask_count_path = os.path.join(zarr_output_dir, f"mask_count_part_{part_id:03d}.zarr")

            if not os.path.exists(mask_pred_path) or not os.path.exists(mask_count_path):
                logger.warning(f"Missing partition {part_id} at {zarr_output_dir}, skipping")
                continue

            logger.info(f"Loading partition {part_id}/{num_parts}")
            mask_pred_z = zarr.open(mask_pred_path, mode='r')
            mask_count_z = zarr.open(mask_count_path, mode='r')

            # Process in chunks to avoid loading entire partition into memory
            for y in range(0, H, chunk_size):
                y_end = min(y + chunk_size, H)
                for x in range(0, W, chunk_size):
                    x_end = min(x + chunk_size, W)

                    # Read chunks and accumulate
                    pred_chunk = mask_pred_z[y:y_end, x:x_end]
                    count_chunk = mask_count_z[y:y_end, x:x_end]

                    total_pred[y:y_end, x:x_end] += pred_chunk
                    total_count[y:y_end, x:x_end] += count_chunk

            logger.info(f"Completed partition {part_id}/{num_parts}")

        # Blend: divide total_pred by total_count
        logger.info("Blending accumulated predictions")
        result = total_pred / np.clip(total_count, 1e-6, None)
        result = np.clip(result, 0, 1)

        logger.info(f"Reduce completed. Final prediction shape: {result.shape}")
        return result

    except Exception as e:
        logger.error(f"Error in reduce_partitions: {e}")
        raise

def load_model(model_path: str, device: torch.device) -> RegressionPLModel:
    """
    Load and initialize the TimeSformer model.

    Args:
        model_path: Path to model checkpoint
        device: Torch device to load model onto

    Returns:
        Loaded and initialized model
    """
    try:
        logger.info(f"Loading model from: {model_path}")

        # Try to load with PyTorch Lightning first
        try:
            model = RegressionPLModel.load_from_checkpoint(model_path, strict=False)
            logger.info("Model loaded with PyTorch Lightning")
        except Exception as e:
            logger.warning(f"PyTorch Lightning loading failed: {e}, trying manual loading")
            # Fallback to manual loading
            model = RegressionPLModel(pred_shape=(1, 1))
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            logger.info("Model loaded manually")

        # Setup multi-GPU if available
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
            logger.info(f"Model wrapped with DataParallel for {torch.cuda.device_count()} GPUs")

        # Move to device
        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully on {device}")
        return model

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def write_tiled_tiff(prediction: np.ndarray, output_path: str) -> None:
    """
    Write prediction array to a tiled TIFF file.

    Args:
        prediction: Float32 array with values in [0, 1], shape (H, W)
        output_path: Path to output TIFF file
    """
    try:
        logger.info(f"Writing tiled TIFF to {output_path}")

        # Convert float32 [0, 1] to uint8 [0, 255]
        pred_uint8 = (np.clip(prediction, 0, 1) * 255).astype(np.uint8)

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Write with tiling and compression
        tiff.imwrite(
            output_path,
            pred_uint8,
            compression='deflate',
            tile=(256, 256),
            metadata={'software': 'optimized_inference'}
        )

        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        logger.info(f"Wrote tiled TIFF: {output_path} ({file_size_mb:.2f} MB)")

    except Exception as e:
        logger.error(f"Error writing tiled TIFF: {e}")
        raise

def run_inference(
    layers: Union[np.ndarray, str],
    model: RegressionPLModel,
    device: torch.device,
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False
) -> Union[np.ndarray, Dict[str, str]]:
    """
    Main entrypoint: accepts either a stacked array (H,W,C) or path to a zarr array.

    Args:
        layers: Either numpy array (H, W, C) or path to zarr array
        model: The inference model
        device: Torch device
        fragment_mask: Optional mask array (H, W)
        is_reverse_segment: Whether to reverse layer order

    Returns:
        - If num_parts == 1: returns (H,W) heatmap numpy array
        - If num_parts > 1: returns dict with zarr paths {"mask_pred": path, "mask_count": path}
    """
    try:
        logger.info("Starting inference process...")
        source, mask, orig_shape, reverse = preprocess_layers(
            layers, fragment_mask, is_reverse_segment
        )
        test_loader, pred_shape, partition_info = create_inference_dataloader(source, mask, reverse)
        result = predict_fn(test_loader, model, device, pred_shape)

        # If partitioned mode, return zarr paths directly
        if CFG.num_parts > 1:
            logger.info("Inference partition completed successfully")
            return result

        # Standard mode: crop to original size and re-normalize per-fragment
        mask_pred = result  # result is np.ndarray in this case
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
