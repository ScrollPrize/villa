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
import time
from typing import List, Tuple, Optional, Union, Dict, Protocol
from contextlib import suppress

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
from profiling import get_worker_profiler, scoped_timer
from processing import path_exists, get_cached_zarr_store

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


def _downsample_mean_last_axis_factor2(current: np.ndarray) -> np.ndarray:
    out_c = int((current.shape[2] + 1) // 2)
    accum = current[..., 0::2].astype(np.float64, copy=True)
    counts = np.ones(out_c, dtype=np.float64)
    odd = current[..., 1::2]
    if odd.shape[2] > 0:
        accum[..., :odd.shape[2]] += odd.astype(np.float64, copy=False)
        counts[:odd.shape[2]] += 1.0
    mean = accum / counts.reshape((1, 1, -1))
    if np.issubdtype(current.dtype, np.integer):
        mean = np.rint(mean).astype(current.dtype, copy=False)
    else:
        mean = mean.astype(current.dtype, copy=False)
    return np.ascontiguousarray(mean)


def _resolve_z_window(
    *,
    full_depth: int,
    start_z: Optional[int],
    end_z: Optional[int],
    z_downsample_mean_factor: int,
    target_in_chans: Optional[int],
) -> Tuple[int, int, int]:
    raw_start = int(start_z if start_z is not None else 0)
    raw_end = int(end_z if end_z is not None else full_depth)

    if raw_end > full_depth:
        logger.warning(f"Requested end_z={raw_end} exceeds available channels={full_depth}, clamping")
        raw_end = full_depth

    if raw_start >= raw_end:
        raise ValueError(f"Invalid z-range: start_z={raw_start} >= end_z={raw_end}")

    raw_count = int(raw_end - raw_start)
    original_raw_start = raw_start
    original_raw_end = raw_end
    if z_downsample_mean_factor == 1:
        output_count = raw_count
        if target_in_chans is not None:
            target = int(target_in_chans)
            if target < 1 or target > output_count:
                raise ValueError(
                    f"target_in_chans must be in [1, {output_count}] for z_downsample_mean_factor=1, got {target}"
                )
            crop_start = max(0, (output_count - target) // 2)
            raw_start += crop_start
            raw_end = raw_start + target
            output_count = target
        return raw_start, raw_end, output_count

    if z_downsample_mean_factor != 2:
        raise ValueError(f"Unsupported z_downsample_mean_factor={z_downsample_mean_factor}; expected 1 or 2")

    downsampled_count = int((raw_count + z_downsample_mean_factor - 1) // z_downsample_mean_factor)
    target = int(target_in_chans) if target_in_chans is not None else downsampled_count
    if target < 1 or target > downsampled_count:
        raise ValueError(
            f"target_in_chans must be in [1, {downsampled_count}] after z downsample x{z_downsample_mean_factor}, got {target}"
        )

    cropped_start = max(0, (downsampled_count - target) // 2)
    cropped_stop = int(cropped_start + target)
    raw_start = int(original_raw_start + cropped_start * z_downsample_mean_factor)
    raw_end = int(
        min(
            original_raw_end,
            original_raw_start + cropped_stop * z_downsample_mean_factor,
        )
    )
    raw_end = min(raw_end, full_depth)

    output_count = int((max(0, raw_end - raw_start) + z_downsample_mean_factor - 1) // z_downsample_mean_factor)
    if output_count != target:
        raise ValueError(
            f"Expected {target} channels after z downsample x{z_downsample_mean_factor}, got {output_count} from raw slice [{raw_start}, {raw_end})"
        )
    return raw_start, raw_end, output_count


def _resolve_xy_roi(
    *,
    full_h: int,
    full_w: int,
    roi_xyxy: Optional[Tuple[int, int, int, int]],
) -> Tuple[int, int, int, int]:
    if roi_xyxy is None:
        return 0, 0, full_w, full_h

    x0, y0, x1, y1 = [int(v) for v in roi_xyxy]
    x0 = max(0, min(full_w, x0))
    y0 = max(0, min(full_h, y0))
    x1 = max(0, min(full_w, x1))
    y1 = max(0, min(full_h, y1))
    if x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"ROI must intersect the source volume; got clamped ROI "
            f"[{x0}, {y0}, {x1}, {y1}) for full shape {full_h}x{full_w}"
        )
    return x0, y0, x1, y1

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
    accumulator_mode: str = "ram"  # "auto", "ram", or "memmap"
    accumulator_auto_threshold_gib: float = 32.0
    accumulator_flush_every_batches: int = 64
    progress_log_every_batches: int = 0
    progress_log_every_seconds: float = 0.0
    max_batches: int = 0
    skip_partition_write: bool = False
    skip_empty_tiles: bool = True
    prune_empty_tiles: bool = False

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
    direct_single_part_write: bool = False

    # Compression settings
    use_zarr_compression: bool = True

CFG = InferenceConfig()


def _use_memmap_accumulators(h: int, w: int) -> bool:
    mode = str(getattr(CFG, "accumulator_mode", "auto")).strip().lower()
    if mode == "memmap":
        return True
    if mode == "ram":
        return False
    if mode != "auto":
        raise ValueError(f"Unsupported accumulator_mode={mode!r}; expected 'auto', 'ram', or 'memmap'")
    total_bytes = int(h) * int(w) * np.dtype(np.float32).itemsize * 2
    threshold_bytes = float(getattr(CFG, "accumulator_auto_threshold_gib", 32.0)) * (1024 ** 3)
    return total_bytes >= threshold_bytes


def _make_accumulators(h: int, w: int) -> Tuple[np.ndarray, np.ndarray, bool, List[str]]:
    use_memmap = _use_memmap_accumulators(h, w)
    if not use_memmap:
        return (
            np.zeros((h, w), dtype=np.float32),
            np.zeros((h, w), dtype=np.float32),
            False,
            [],
        )

    runtime_dir = os.environ.get("RUNTIME_DIR", CFG.zarr_output_dir)
    os.makedirs(runtime_dir, exist_ok=True)
    pred_tmp = os.path.join(runtime_dir, f"mask_pred_accum_part_{CFG.part_id:03d}.f32")
    count_tmp = os.path.join(runtime_dir, f"mask_count_accum_part_{CFG.part_id:03d}.f32")
    for path in (pred_tmp, count_tmp):
        with suppress(FileNotFoundError):
            os.remove(path)

    mask_pred = np.memmap(pred_tmp, mode="w+", dtype=np.float32, shape=(h, w))
    mask_count = np.memmap(count_tmp, mode="w+", dtype=np.float32, shape=(h, w))
    mask_pred[:] = 0.0
    mask_count[:] = 0.0
    logger.info(
        "Using disk-backed accumulators in %s for prediction canvas %sx%s "
        "(two float32 planes ~= %.2f GiB)",
        runtime_dir,
        h,
        w,
        (int(h) * int(w) * np.dtype(np.float32).itemsize * 2) / (1024 ** 3),
    )
    return mask_pred, mask_count, True, [pred_tmp, count_tmp]


def _write_array_to_zarr_chunks(
    src: np.ndarray,
    out_path: str,
    shape: Tuple[int, int],
    compressor,
    profiler=None,
    chunk_size: int = 1024,
) -> None:
    dst = zarr.open(
        out_path,
        mode='w',
        shape=shape,
        chunks=(chunk_size, chunk_size),
        dtype=np.float32,
        compressor=compressor,
        zarr_format=2,
        config={'write_empty_chunks': False}
    )
    h, w = shape
    with scoped_timer(profiler, "zarr_write_seconds", flag="approximate"):
        for y0 in range(0, h, chunk_size):
            y1 = min(y0 + chunk_size, h)
            for x0 in range(0, w, chunk_size):
                x1 = min(x0 + chunk_size, w)
                dst[y0:y1, x0:x1] = np.asarray(src[y0:y1, x0:x1], dtype=np.float32)

# --------------------- Disk-backed / Array-backed layers ---------------------
class LayersSource:
    """
    Unified source of layers that supports:
      • numpy arrays of shape (H, W, C)
      • path to an existing zarr array

    For reading tiles during inference without holding the whole stack in RAM.
    """
    def __init__(
        self,
        src: Union[np.ndarray, str],
        start_z: Optional[int] = None,
        end_z: Optional[int] = None,
        z_downsample_mean_factor: int = 1,
        target_in_chans: Optional[int] = None,
        roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    ):
        self._z_downsample_mean_factor = int(z_downsample_mean_factor or 1)
        if self._z_downsample_mean_factor not in (1, 2):
            raise ValueError(
                f"Unsupported z_downsample_mean_factor={self._z_downsample_mean_factor}; expected 1 or 2"
            )
        if isinstance(src, np.ndarray):
            if src.ndim != 3:
                raise ValueError(f"Expected (H,W,C) array, got {src.shape}")
            self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1 = _resolve_xy_roi(
                full_h=int(src.shape[0]),
                full_w=int(src.shape[1]),
                roi_xyxy=roi_xyxy,
            )
            start_z = start_z if start_z is not None else 0
            end_z = end_z if end_z is not None else src.shape[2]
            raw_start, raw_end, output_count = _resolve_z_window(
                full_depth=int(src.shape[2]),
                start_z=start_z,
                end_z=end_z,
                z_downsample_mean_factor=self._z_downsample_mean_factor,
                target_in_chans=target_in_chans,
            )
            current = src[self._roi_y0:self._roi_y1, self._roi_x0:self._roi_x1, raw_start:raw_end]
            if self._z_downsample_mean_factor == 2:
                current = _downsample_mean_last_axis_factor2(current)
            self._arr = current
            self._mm = None
            self._shape = (self._arr.shape[0], self._arr.shape[1], output_count)
            self._dtype = self._arr.dtype
            self._needs_transpose = False
            self._start_z = raw_start
            self._end_z = raw_end
            self.storage_kind = "array"
        elif isinstance(src, str):
            if not path_exists(src):
                raise ValueError(f"Zarr path does not exist: {src}")
            self._arr = None
            store = get_cached_zarr_store(src)
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
            self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1 = _resolve_xy_roi(
                full_h=int(full_shape[0]),
                full_w=int(full_shape[1]),
                roi_xyxy=roi_xyxy,
            )

            self._start_z, self._end_z, output_count = _resolve_z_window(
                full_depth=int(full_shape[2]),
                start_z=start_z,
                end_z=end_z,
                z_downsample_mean_factor=self._z_downsample_mean_factor,
                target_in_chans=target_in_chans,
            )
            self.storage_kind = "remote_zarr" if src.startswith("s3://") else "local_zarr"
            self._shape = (
                self._roi_y1 - self._roi_y0,
                self._roi_x1 - self._roi_x0,
                output_count,
            )
            logger.info(
                f"Loaded zarr: shape={self._shape}, roi=[{self._roi_x0}, {self._roi_y0}, {self._roi_x1}, {self._roi_y1}), "
                f"raw_z_range=[{self._start_z}, {self._end_z}), "
                f"z_downsample_mean_factor={self._z_downsample_mean_factor}, dtype={self._dtype}"
            )
        else:
            raise TypeError("LayersSource expects np.ndarray or str (zarr path)")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return self._shape

    @property
    def roi_xyxy(self) -> Tuple[int, int, int, int]:
        return (self._roi_x0, self._roi_y0, self._roi_x1, self._roi_y1)

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
                src_y1 = self._roi_y0 + yy1
                src_y2 = self._roi_y0 + yy2
                src_x1 = self._roi_x0 + xx1
                src_x2 = self._roi_x0 + xx2
                if self._needs_transpose:
                    roi = self._mm[self._start_z:self._end_z, src_y1:src_y2, src_x1:src_x2]
                    roi = np.transpose(roi, (1, 2, 0))
                else:
                    roi = self._mm[src_y1:src_y2, src_x1:src_x2, self._start_z:self._end_z]
                if self._z_downsample_mean_factor == 2:
                    roi = _downsample_mean_last_axis_factor2(np.asarray(roi))
                out[(yy1 - y1):(yy2 - y1), (xx1 - x1):(xx2 - x1)] = roi
        return out

    def roi_has_signal(self, y1: int, y2: int, x1: int, x2: int) -> bool:
        """
        Return True when the in-bounds portion of the ROI contains any non-zero voxel.

        This is used to prune empty tiles before building the dataloader so they never
        contribute to dataset length or inference scheduling.
        """
        H, W, _ = self._shape
        yy1, yy2 = max(0, y1), min(H, y2)
        xx1, xx2 = max(0, x1), min(W, x2)
        if yy2 <= yy1 or xx2 <= xx1:
            return False

        if self._arr is not None:
            roi = self._arr[yy1:yy2, xx1:xx2, :]
        else:
            src_y1 = self._roi_y0 + yy1
            src_y2 = self._roi_y0 + yy2
            src_x1 = self._roi_x0 + xx1
            src_x2 = self._roi_x0 + xx2
            if self._needs_transpose:
                roi = self._mm[self._start_z:self._end_z, src_y1:src_y2, src_x1:src_x2]
                roi = np.transpose(roi, (1, 2, 0))
            else:
                roi = self._mm[src_y1:src_y2, src_x1:src_x2, self._start_z:self._end_z]
            if self._z_downsample_mean_factor == 2:
                roi = _downsample_mean_last_axis_factor2(np.asarray(roi))

        return bool(np.any(roi))

# ----------------------------- Preprocess ------------------------------------
def preprocess_layers(
    layers: Union[np.ndarray, str],
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False,
    start_z: Optional[int] = None,
    end_z: Optional[int] = None,
    z_downsample_mean_factor: int = 1,
    target_in_chans: Optional[int] = None,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
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
        src = LayersSource(
            layers,
            start_z=start_z,
            end_z=end_z,
            z_downsample_mean_factor=z_downsample_mean_factor,
            target_in_chans=target_in_chans,
            roi_xyxy=roi_xyxy,
        )
        h, w, c = src.shape
        if c != CFG.in_chans:
            logger.warning(f"Model expects {CFG.in_chans} channels, got {c}")

        if fragment_mask is None:
            fragment_mask = np.ones((h, w), dtype=np.uint8) * 255
        else:
            roi_x0, roi_y0, roi_x1, roi_y1 = src.roi_xyxy
            fragment_mask = fragment_mask[roi_y0:roi_y1, roi_x0:roi_x1]

        logger.info(
            f"Prepared layers source: shape={src.shape}, roi={src.roi_xyxy}, "
            f"mask={fragment_mask.shape}, reverse={is_reverse_segment}"
        )
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
        worker_profiler = get_worker_profiler()
        with scoped_timer(worker_profiler, "preprocess_seconds", flag="approximate"):
            x1, y1, x2, y2 = self.xyxys[idx].tolist()
            read_metric = "local_read_seconds"
            if self.source.storage_kind == "remote_zarr":
                read_metric = "remote_read_seconds"
            with scoped_timer(worker_profiler, read_metric, flag="approximate"):
                tile = self.source.read_roi(y1, y2, x1, x2)  # (tile, tile, C), uint8
            if self.source.storage_kind != "remote_zarr" and worker_profiler is not None:
                worker_profiler.increment_counter("local_read_bytes", int(tile.nbytes), flag="approximate")
            if self.reverse:
                tile = tile[:, :, ::-1]
            # Clip to match training range - in-place for speed
            np.clip(tile, 0, CFG.max_clip_value, out=tile)
            is_empty = bool(CFG.skip_empty_tiles and tile.max() == 0)

            data = self.transform(image=tile)  # -> tensor (C,H,W)
            tens = data["image"].unsqueeze(0)  # -> (1,C,H,W) so C becomes frames
            return tens, self.xyxys[idx], is_empty

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

        total_grid_tiles = len(x1_list) * len(y1_list)
        xyxys: List[List[int]] = []
        mask_filtered_tiles = 0
        empty_pruned_tiles = 0
        tile_prune_start = time.monotonic()
        if CFG.prune_empty_tiles:
            logger.info(
                "Pruning empty tiles at tile-list creation time across %d candidates",
                total_grid_tiles,
            )
        for y1 in y1_list:
            for x1 in x1_list:
                y2, x2 = y1 + CFG.tile_size, x1 + CFG.tile_size
                # compute valid ratio inside bounds
                yy1, yy2 = max(0, y1), min(h, y2)
                xx1, xx2 = max(0, x1), min(w, x2)
                roi = fragment_mask[yy1:yy2, xx1:xx2]
                valid_ratio = float(roi.size and (roi != 0).mean() or 0.0)
                if valid_ratio < CFG.min_valid_ratio:
                    mask_filtered_tiles += 1
                    continue
                if CFG.prune_empty_tiles and not source.roi_has_signal(y1, y2, x1, x2):
                    empty_pruned_tiles += 1
                    continue
                xyxys.append([x1, y1, x2, y2])

        if not xyxys:
            raise ValueError("No valid tiles (mask empty or fully filtered).")

        total_tiles = len(xyxys)

        # Apply range-based partitioning if num_parts > 1
        partition_info = {
            "grid_tiles": total_grid_tiles,
            "total_tiles": total_tiles,
            "start_idx": 0,
            "end_idx": total_tiles,
            "partition_tiles": total_tiles,
            "mask_filtered_tiles": mask_filtered_tiles,
            "empty_pruned_tiles": empty_pruned_tiles,
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
        logger.info(
            "Created dataloader with %d tiles (grid=%d, mask_filtered=%d, empty_pruned=%d, prune_seconds=%.2f)",
            len(dataset),
            total_grid_tiles,
            mask_filtered_tiles,
            empty_pruned_tiles,
            time.monotonic() - tile_prune_start,
        )
        return loader, (h, w), partition_info

    except Exception as e:
        logger.error(f"Error creating dataloader: {e}")
        raise

# ----------------------------- Inference -------------------------------------
def predict_fn(
    test_loader: DataLoader,
    model: InferenceModel,
    device: torch.device,
    pred_shape: Tuple[int, int],
    profiler=None,
) -> Dict[str, str]:
    """
    Run tiled inference and write results to zarr files.

    Returns:
        dict with zarr paths {"mask_pred": path, "mask_count": path}
    """
    try:
        H, W = pred_shape

        accumulator_start = time.monotonic()
        mask_pred, mask_count, using_memmap, accumulator_paths = _make_accumulators(H, W)
        logger.info(
            "Accumulator setup completed in %.2f seconds (mode=%s)",
            time.monotonic() - accumulator_start,
            "memmap" if using_memmap else "ram",
        )
        flush_every = max(0, int(getattr(CFG, "accumulator_flush_every_batches", 64) or 0))
        progress_log_every_batches = max(0, int(getattr(CFG, "progress_log_every_batches", 0) or 0))
        progress_log_every_seconds = float(getattr(CFG, "progress_log_every_seconds", 0.0) or 0.0)
        max_batches = max(0, int(getattr(CFG, "max_batches", 0) or 0))
        skip_partition_write = bool(getattr(CFG, "skip_partition_write", False))

        weight_tensor: Optional[torch.Tensor] = None
        model.eval()
        batch_count = 0
        tile_count = 0
        inferred_tile_count = 0
        skipped_empty_tile_count = 0
        torch_profiler_batches = 20
        if profiler is not None and profiler.enable_torch_profiler():
            profiler.start_torch_profiler(use_cuda=device.type == "cuda")

        with torch.inference_mode():
            try:
                total_tiles = len(test_loader.dataset)
            except Exception:
                total_tiles = None
            infer_wall_start = time.monotonic()
            last_progress_log_time = infer_wall_start
            last_progress_log_tiles = 0
            last_progress_log_batches = 0
            pbar = tqdm(total=total_tiles,
                        desc="Running inference",
                        unit="tile",
                        **get_tqdm_kwargs())

            for (images, xys, is_empty_batch) in test_loader:
                batch_count += 1
                batch_tiles = int(images.size(0))
                tile_count += batch_tiles
                if torch.is_tensor(is_empty_batch):
                    non_empty_mask = ~is_empty_batch.bool()
                else:
                    non_empty_mask = torch.as_tensor(
                        np.logical_not(np.asarray(is_empty_batch, dtype=bool)),
                        dtype=torch.bool,
                    )
                non_empty_tiles = int(non_empty_mask.sum().item())
                skipped_now = batch_tiles - non_empty_tiles
                skipped_empty_tile_count += skipped_now
                if non_empty_tiles == 0:
                    now = time.monotonic()
                    should_log_progress = False
                    if progress_log_every_batches > 0 and batch_count % progress_log_every_batches == 0:
                        should_log_progress = True
                    if progress_log_every_seconds > 0 and (now - last_progress_log_time) >= progress_log_every_seconds:
                        should_log_progress = True
                    if should_log_progress:
                        elapsed = max(now - infer_wall_start, 1e-6)
                        window_elapsed = max(now - last_progress_log_time, 1e-6)
                        batch_delta = batch_count - last_progress_log_batches
                        tile_delta = tile_count - last_progress_log_tiles
                        completion = (
                            f"{tile_count}/{total_tiles} ({100.0 * tile_count / total_tiles:.2f}%)"
                            if total_tiles
                            else str(tile_count)
                        )
                        logger.info(
                            "Inference progress: scheduled_tiles=%s inferred_tiles=%d skipped_empty_tiles=%d batches=%d "
                            "avg_tiles_per_sec=%.2f window_tiles_per_sec=%.2f avg_batches_per_sec=%.3f window_batches_per_sec=%.3f",
                            completion,
                            inferred_tile_count,
                            skipped_empty_tile_count,
                            batch_count,
                            tile_count / elapsed,
                            tile_delta / window_elapsed,
                            batch_count / elapsed,
                            batch_delta / window_elapsed,
                        )
                        last_progress_log_time = now
                        last_progress_log_tiles = tile_count
                        last_progress_log_batches = batch_count
                    pbar.update(batch_tiles)
                    if max_batches > 0 and batch_count >= max_batches:
                        logger.info(
                            "Reached max_batches=%d after %d batches / %d scheduled tiles; stopping early for benchmark mode",
                            max_batches,
                            batch_count,
                            tile_count,
                        )
                        break
                    continue
                if non_empty_tiles != batch_tiles:
                    images = images[non_empty_mask]
                    if torch.is_tensor(xys):
                        xys = xys[non_empty_mask]
                    else:
                        xys = np.asarray(xys)[non_empty_mask.cpu().numpy()]
                inferred_tile_count += non_empty_tiles
                with scoped_timer(profiler, "host_to_device_seconds", cuda_sync=device.type == "cuda"):
                    images = images.to(device, non_blocking=True)

                amp_device = "cuda" if device.type == "cuda" else "cpu"
                with scoped_timer(profiler, "forward_seconds", cuda_sync=device.type == "cuda"):
                    with torch.autocast(device_type=amp_device, enabled=True):
                        y_preds = model.forward(images)  # Model-specific forward
                    y_preds = torch.sigmoid(y_preds)
                    y_preds_resized = F.interpolate(
                        y_preds.float(),
                        size=(CFG.tile_size, CFG.tile_size),
                        mode='bilinear',
                        align_corners=False
                    )  # (B,1,tile,tile)
                    if profiler is not None and profiler._torch_profiler is not None and batch_count <= torch_profiler_batches:
                        with suppress(Exception):
                            profiler._torch_profiler.step()
                        if batch_count == torch_profiler_batches:
                            profiler.stop_torch_profiler()

                # Get scale factor from model
                scale_factor = model.get_output_scale_factor()

                if weight_tensor is None:
                    th, tw = y_preds_resized.shape[-2:]
                    if CFG.use_hann_window:
                        w_np = hann2d(th, tw).astype(np.float32)
                    else:
                        sigma = CFG.gaussian_sigma if CFG.gaussian_sigma > 0 else max(1.0, min(th, tw) / 2.5)
                        w_np = gkern(th, tw, sigma).astype(np.float32)
                    weight_tensor = torch.from_numpy(w_np).to(device)  # (th,tw)

                y_weighted = (y_preds_resized * weight_tensor).squeeze(1)  # (B,th,tw)

                with scoped_timer(profiler, "device_to_host_seconds", cuda_sync=device.type == "cuda"):
                    y_cpu = y_weighted.cpu().numpy()
                    w_cpu = weight_tensor.detach().cpu().numpy().astype(np.float32)

                with scoped_timer(profiler, "postprocess_seconds"):
                    if torch.is_tensor(xys):
                        xys = xys.cpu().numpy().astype(np.int32)
                    for i in range(xys.shape[0]):
                        x1, y1, x2, y2 = [int(v) for v in xys[i]]
                        mask_pred[y1:y2, x1:x2] += y_cpu[i]
                        mask_count[y1:y2, x1:x2] += w_cpu
                    if using_memmap and flush_every > 0 and batch_count % flush_every == 0:
                        mask_pred.flush()
                        mask_count.flush()
                    now = time.monotonic()
                    should_log_progress = False
                    if progress_log_every_batches > 0 and batch_count % progress_log_every_batches == 0:
                        should_log_progress = True
                    if progress_log_every_seconds > 0 and (now - last_progress_log_time) >= progress_log_every_seconds:
                        should_log_progress = True
                    if should_log_progress:
                        elapsed = max(now - infer_wall_start, 1e-6)
                        window_elapsed = max(now - last_progress_log_time, 1e-6)
                        batch_delta = batch_count - last_progress_log_batches
                        tile_delta = tile_count - last_progress_log_tiles
                        completion = (
                            f"{tile_count}/{total_tiles} ({100.0 * tile_count / total_tiles:.2f}%)"
                            if total_tiles
                            else str(tile_count)
                        )
                        logger.info(
                            "Inference progress: scheduled_tiles=%s inferred_tiles=%d skipped_empty_tiles=%d batches=%d avg_tiles_per_sec=%.2f "
                            "window_tiles_per_sec=%.2f avg_batches_per_sec=%.3f window_batches_per_sec=%.3f",
                            completion,
                            inferred_tile_count,
                            skipped_empty_tile_count,
                            batch_count,
                            tile_count / elapsed,
                            tile_delta / window_elapsed,
                            batch_count / elapsed,
                            batch_delta / window_elapsed,
                        )
                        last_progress_log_time = now
                        last_progress_log_tiles = tile_count
                        last_progress_log_batches = batch_count
                pbar.update(batch_tiles)
                if max_batches > 0 and batch_count >= max_batches:
                    logger.info(
                        "Reached max_batches=%d after %d batches / %d scheduled tiles; stopping early for benchmark mode",
                        max_batches,
                        batch_count,
                        tile_count,
                    )
                    break
            pbar.close()

        if profiler is not None:
            profiler.set_metric("partition_batches", batch_count, semantics="counter delta")
            profiler.set_metric("partition_tiles", tile_count, semantics="counter delta")
            profiler.set_metric("partition_inferred_tiles", inferred_tile_count, semantics="counter delta")
            profiler.set_metric("partition_skipped_empty_tiles", skipped_empty_tile_count, semantics="counter delta")
            if batch_count > 0:
                active_seconds = (
                    float(profiler.metrics.get("host_to_device_seconds") or 0.0)
                    + float(profiler.metrics.get("forward_seconds") or 0.0)
                    + float(profiler.metrics.get("device_to_host_seconds") or 0.0)
                    + float(profiler.metrics.get("postprocess_seconds") or 0.0)
                )
                if active_seconds > 0:
                    profiler.set_metric(
                        "steady_state_tiles_per_second",
                        float(tile_count) / active_seconds,
                        flag="estimated",
                    )

        if skip_partition_write:
            logger.info(
                "Skipping partition zarr write for partition %d (benchmark mode enabled)",
                CFG.part_id,
            )
            return {
                "mask_pred": "",
                "mask_count": "",
                "partition_batches": batch_count,
                "partition_tiles": tile_count,
                "partition_inferred_tiles": inferred_tile_count,
                "partition_skipped_empty_tiles": skipped_empty_tile_count,
                "skipped_partition_write": True,
            }

        if CFG.direct_single_part_write:
            logger.info(
                "Direct single-part TIFF write enabled; returning in-memory accumulators "
                "for partition %d without spilling partition zarrs",
                CFG.part_id,
            )
            return {
                "mask_pred_array": mask_pred,
                "mask_count_array": mask_count,
                "partition_batches": batch_count,
                "partition_tiles": tile_count,
                "partition_inferred_tiles": inferred_tile_count,
                "partition_skipped_empty_tiles": skipped_empty_tile_count,
                "direct_single_part_write": True,
            }

        from numcodecs import LZ4

        logger.info(f"Writing partition {CFG.part_id} results to zarr arrays...")
        os.makedirs(CFG.zarr_output_dir, exist_ok=True)

        mask_pred_path = os.path.join(CFG.zarr_output_dir, f"mask_pred_part_{CFG.part_id:03d}.zarr")
        mask_count_path = os.path.join(CFG.zarr_output_dir, f"mask_count_part_{CFG.part_id:03d}.zarr")

        compressor = LZ4(acceleration=1) if CFG.use_zarr_compression else None

        if using_memmap:
            mask_pred.flush()
            mask_count.flush()

        _write_array_to_zarr_chunks(mask_pred, mask_pred_path, (H, W), compressor, profiler=profiler)
        _write_array_to_zarr_chunks(mask_count, mask_count_path, (H, W), compressor, profiler=profiler)
        if profiler is not None:
            profiler.increment_counter("local_write_bytes", int(mask_pred.nbytes + mask_count.nbytes), flag="approximate")

        logger.info(f"Partition {CFG.part_id} completed. Wrote zarr arrays to {CFG.zarr_output_dir}")
        return {
            "mask_pred": mask_pred_path,
            "mask_count": mask_count_path,
            "partition_batches": batch_count,
            "partition_tiles": tile_count,
            "partition_inferred_tiles": inferred_tile_count,
            "partition_skipped_empty_tiles": skipped_empty_tile_count,
        }

    except Exception as e:
        logger.error(f"Error in predict_fn: {e}")
        raise
    finally:
        for arr in ("mask_pred", "mask_count"):
            value = locals().get(arr)
            with suppress(Exception):
                if isinstance(value, np.memmap):
                    value.flush()
        for path in locals().get("accumulator_paths", []):
            with suppress(FileNotFoundError):
                os.remove(path)


def run_inference(
    layers: Union[np.ndarray, str],
    model: InferenceModel,
    device: torch.device,
    fragment_mask: Optional[np.ndarray] = None,
    is_reverse_segment: bool = False,
    start_z: Optional[int] = None,
    end_z: Optional[int] = None,
    z_downsample_mean_factor: int = 1,
    target_in_chans: Optional[int] = None,
    roi_xyxy: Optional[Tuple[int, int, int, int]] = None,
    profiler=None,
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
        with scoped_timer(profiler, "preprocess_seconds", flag="approximate"):
            source, mask, orig_shape, reverse = preprocess_layers(
                layers,
                fragment_mask,
                is_reverse_segment,
                start_z=start_z,
                end_z=end_z,
                z_downsample_mean_factor=z_downsample_mean_factor,
                target_in_chans=target_in_chans,
                roi_xyxy=roi_xyxy,
            )
        test_loader, pred_shape, partition_info = create_inference_dataloader(source, mask, reverse)
        if profiler is not None:
            profiler.set_metric("partition_tiles", partition_info.get("partition_tiles", 0), semantics="counter delta")
            profiler.set_metric("tiles_total", partition_info.get("total_tiles", 0), semantics="metadata")
        result = predict_fn(test_loader, model, device, pred_shape, profiler=profiler)
        if profiler is not None:
            result["partition_info"] = partition_info

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
