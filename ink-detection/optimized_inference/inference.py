"""
Common inference infrastructure for ink detection models.
Supports both TimeSformer and ResNet3D architectures.

Key features:
- Memory-efficient streaming from zarr or numpy arrays
- Sliding window inference with overlap-add blending
- Partitioned inference for distributed processing
- Model-agnostic pipeline
"""
import os
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", "0")
import gc
import math
import logging
from typing import List, Tuple, Optional, Union, Dict, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.auto import tqdm
import zarr

from k8s import get_tqdm_kwargs
from processing import path_exists, get_zarr_store

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

# ----------------------------- Model Protocol --------------------------------
class InferenceModel(Protocol):
    """Protocol that model wrappers must implement."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on input tensor."""
        ...

    def get_output_scale_factor(self) -> int:
        """Get the scale factor for interpolating model output to tile size."""
        ...

    def eval(self):
        """Set model to evaluation mode."""
        ...

    def to(self, device: torch.device):
        """Move model to device."""
        ...

# ----------------------------- Config ----------------------------------------
class InferenceConfig:
    # Model configuration
    model_type: str = "timesformer"  # "timesformer" or "resnet3d"
    in_chans: int = 26  # Will be overridden by model-specific defaults
    encoder_depth: int = 5

    # Inference configuration
    size: int = 64       # net input size (post-resize, normally equals tile_size)
    tile_size: int = 64
    stride: int = 16
    batch_size: int = 256
    workers: int = min(8, os.cpu_count() or 4)
    prefetch_factor: int = 8

    # Image processing / scaling
    max_clip_value: int = 200

    # Blending
    use_hann_window: bool = True
    gaussian_sigma: float = 0.0  # if using Gaussian: 0 -> auto (~ tile/2.5)

    # Tile selection
    min_valid_ratio: float = 0.0

    # Partitioning for map/reduce inference
    num_parts: int = 1
    part_id: int = 0
    zarr_output_dir: str = os.environ.get("ZARR_OUTPUT_DIR", "/tmp/partitions")

    # Compression settings
    use_zarr_compression: bool = True

CFG = InferenceConfig()

# --------------------- Disk-backed / Array-backed layers ---------------------
class LayersSource:
    """
    Unified source of layers that supports:
      • numpy arrays of shape (H, W, C)
      • path to an existing zarr array

    For reading tiles during inference without holding the whole stack in RAM.
    """
    def __init__(self, src: Union[np.ndarray, str], start_z: Optional[int] = None, end_z: Optional[int] = None):
        if isinstance(src, np.ndarray):
            if src.ndim != 3:
                raise ValueError(f"Expected (H,W,C) array, got {src.shape}")
            start_z = start_z if start_z is not None else 0
            end_z = end_z if end_z is not None else src.shape[2]
            self._arr = src[:, :, start_z:end_z]
            self._mm = None
            self._shape = self._arr.shape
            self._dtype = self._arr.dtype
            self._needs_transpose = False
            self._start_z = None
            self._end_z = None
        elif isinstance(src, str):
            if not path_exists(src):
                raise ValueError(f"Zarr path does not exist: {src}")
            self._arr = None
            store = get_zarr_store(src)
            root = zarr.open(store, mode='r')

            if isinstance(root, zarr.Group):
                if "0" in root:
                    self._mm = root["0"]
                else:
                    raise ValueError(f"OME-Zarr group found but no '0' array present. Available keys: {list(root.keys())}")
            else:
                self._mm = root

            raw_shape = self._mm.shape
            self._dtype = self._mm.dtype
            if len(raw_shape) != 3:
                raise ValueError(f"Expected 3D zarr, got shape {raw_shape}")

            # we support depth dimension both as first or last axis
            min_dim_idx = raw_shape.index(min(raw_shape))
            if min_dim_idx == 0:
                self._needs_transpose = True
                full_shape = (raw_shape[1], raw_shape[2], raw_shape[0])
            else:
                self._needs_transpose = False
                full_shape = raw_shape

            self._start_z = start_z if start_z is not None else 0
            self._end_z = end_z if end_z is not None else full_shape[2]

            if self._end_z > full_shape[2]:
                logger.warning(f"Requested end_z={self._end_z} exceeds available channels={full_shape[2]}, clamping")
                self._end_z = full_shape[2]

            if self._start_z >= self._end_z:
                raise ValueError(f"Invalid z-range: start_z={self._start_z} >= end_z={self._end_z}")

            self._shape = (full_shape[0], full_shape[1], self._end_z - self._start_z)
            logger.info(f"Loaded zarr: shape={self._shape}, z-range=[{self._start_z}, {self._end_z}), dtype={self._dtype}")
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
        out = np.zeros((y2 - y1, x2 - x1, C), dtype=self._dtype)
        if yy2 > yy1 and xx2 > xx1:
            if self._arr is not None:
                out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = self._arr[yy1:yy2, xx1:xx2, :]
            else:
                if self._needs_transpose:
                    roi = self._mm[self._start_z:self._end_z, yy1:yy2, xx1:xx2]
                    roi = np.transpose(roi, (1, 2, 0))
                else:
                    roi = self._mm[yy1:yy2, xx1:xx2, self._start_z:self._end_z]
                out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = roi
        return out

# ----------------------------- Preprocess ------------------------------------
def preprocess_layers(
    layers: Union[np.ndarray, str],
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False,
    start_z: Optional[int] = None,
    end_z: Optional[int] = None
) -> Tuple[LayersSource, np.ndarray, Tuple[int, int], bool]:
    """
    Prepare layers for streaming inference.

    Args:
        layers: Either a numpy array (H, W, C) or path to a zarr array
        fragment_mask: Optional mask array (H, W)
        is_reverse_segment: Whether to reverse layer order
        start_z: Optional starting z-layer index (inclusive)
        end_z: Optional ending z-layer index (exclusive)

    Returns:
        (source, mask, orig_shape, reverse_flag)
    """
    try:
        src = LayersSource(layers, start_z=start_z, end_z=end_z)
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
            multiprocessing_context='spawn' if CFG.workers > 0 else None,
        )
        logger.info(f"Created dataloader with {len(dataset)} tiles")
        return loader, (h, w), partition_info

    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise

# ----------------------------- Inference -------------------------------------
def predict_fn(
    test_loader: DataLoader,
    model: InferenceModel,
    device: torch.device,
    pred_shape: Tuple[int, int]
) -> Dict[str, str]:
    """
    Run tiled inference and write results to zarr files.

    Returns:
        dict with zarr paths {"mask_pred": path, "mask_count": path}
    """
    try:
        H, W = pred_shape

        # Always use in-memory numpy arrays during inference loop for speed
        mask_pred = np.zeros((H, W), dtype=np.float32)
        mask_count = np.zeros((H, W), dtype=np.float32)

        weight_tensor: Optional[torch.Tensor] = None
        model.eval()

        with torch.inference_mode():
            try:
                total_tiles = len(test_loader.dataset)
            except Exception:
                total_tiles = None
            pbar = tqdm(total=total_tiles,
                        desc="Running inference",
                        unit="tile",
                        **get_tqdm_kwargs())

            for (images, xys) in test_loader:
                images = images.to(device, non_blocking=True)

                amp_device = "cuda" if device.type == "cuda" else "cpu"
                with torch.autocast(device_type=amp_device, enabled=True):
                    y_preds = model.forward(images)  # Model-specific forward

                y_preds = torch.sigmoid(y_preds)

                # Get scale factor from model
                scale_factor = model.get_output_scale_factor()
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
                pbar.update(images.size(0))
            pbar.close()

        # Always write results to zarr
        from numcodecs import LZ4

        logger.info(f"Writing partition {CFG.part_id} results to zarr arrays...")
        os.makedirs(CFG.zarr_output_dir, exist_ok=True)

        mask_pred_path = os.path.join(CFG.zarr_output_dir, f"mask_pred_part_{CFG.part_id:03d}.zarr")
        mask_count_path = os.path.join(CFG.zarr_output_dir, f"mask_count_part_{CFG.part_id:03d}.zarr")

        compressor = LZ4(acceleration=1) if CFG.use_zarr_compression else None

        # Create and write mask_pred zarr
        mask_pred_z = zarr.open(
            mask_pred_path,
            mode='w',
            shape=(H, W),
            chunks=(1024, 1024),
            dtype=np.float32,
            compressor=compressor,
            zarr_format=2,
            config={'write_empty_chunks': False}
        )
        mask_pred_z[:] = mask_pred

        # Create and write mask_count zarr
        mask_count_z = zarr.open(
            mask_count_path,
            mode='w',
            shape=(H, W),
            chunks=(1024, 1024),
            dtype=np.float32,
            compressor=compressor,
            zarr_format=2,
            config={'write_empty_chunks': False}
        )
        mask_count_z[:] = mask_count

        logger.info(f"Partition {CFG.part_id} completed. Wrote zarr arrays to {CFG.zarr_output_dir}")
        return {
            "mask_pred": mask_pred_path,
            "mask_count": mask_count_path,
        }

    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        raise


def run_inference(
    layers: Union[np.ndarray, str],
    model: InferenceModel,
    device: torch.device,
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False,
    start_z: Optional[int] = None,
    end_z: Optional[int] = None
) -> Dict[str, str]:
    """
    Main entrypoint: accepts either a stacked array (H,W,C) or path to a zarr array.

    Args:
        layers: Either numpy array (H, W, C) or path to zarr array
        model: The inference model (must implement InferenceModel protocol)
        device: Torch device
        fragment_mask: Optional mask array (H, W)
        is_reverse_segment: Whether to reverse layer order
        start_z: Optional starting z-layer index (inclusive)
        end_z: Optional ending z-layer index (exclusive)

    Returns:
        dict with zarr paths {"mask_pred": path, "mask_count": path}
    """
    try:
        logger.info("Starting inference process...")
        source, mask, orig_shape, reverse = preprocess_layers(
            layers, fragment_mask, is_reverse_segment, start_z=start_z, end_z=end_z
        )
        test_loader, pred_shape, partition_info = create_inference_dataloader(source, mask, reverse)
        result = predict_fn(test_loader, model, device, pred_shape)

        logger.info("Inference completed successfully")
        return result

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
