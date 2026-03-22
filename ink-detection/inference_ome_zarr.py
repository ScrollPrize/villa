#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import logging
import math
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import urlparse

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from monai.data.utils import compute_importance_map
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from tifxyz_dataset.augmentation import compute_equal_length_mirror_axes, iter_mirror_axes


LOGGER = logging.getLogger("inference_ome_zarr")
DEFAULT_OCCUPANCY_SCAN_LEVEL = "3"
DEFAULT_OVERLAP = 0.25
DEFAULT_SW_BATCH_SIZE = 4
DEFAULT_PREFETCH_FACTOR = 2


@dataclass(frozen=True)
class Block:
    y0: int
    x0: int
    valid_h: int
    valid_w: int


@dataclass
class ConfiguredModel:
    model: torch.nn.Module
    roi_size: int
    in_chans: int
    preprocessing: str
    source: str
    amp_dtype: torch.dtype | None = None


@dataclass(frozen=True)
class ChunkKey:
    row: int
    col: int


class TargetHeadWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module, *, target_name: str):
        super().__init__()
        self.model = model
        self.target_name = str(target_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        if not isinstance(outputs, dict):
            raise TypeError(
                f"Expected wrapped model to return a dict of targets, got {type(outputs).__name__}"
            )
        if self.target_name not in outputs:
            raise KeyError(
                f"Missing target {self.target_name!r} in model outputs. "
                f"Available targets: {sorted(outputs.keys())!r}"
            )
        logits = outputs[self.target_name]
        if isinstance(logits, (list, tuple)):
            if len(logits) == 0:
                raise ValueError(f"Target {self.target_name!r} returned an empty logits list")
            logits = logits[0]
        return logits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple MONAI sliding-window inference for OME-Zarr volumes."
    )
    parser.add_argument("input_zarr", nargs="?", help="Input OME-Zarr path or URL.")
    parser.add_argument("checkpoint", type=Path, nargs="?", help="Model checkpoint path.")
    parser.add_argument("output_tiff", type=Path, nargs="?", help="Output uint8 tiled TIFF path.")
    parser.add_argument(
        "--model-type",
        choices=("auto", "resnet3d", "residual_unet", "tifxyz_unet"),
        default="auto",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=None,
        help="Run inference for each segment directory under this folder.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Checkpoint path override (useful with --folder mode).",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional prefix added to folder-mode output TIFF filenames.",
    )
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--resolution", default="0")
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=DEFAULT_PREFETCH_FACTOR,
        help="Number of batches prefetched per worker when --num-workers > 0.",
    )
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--direction",
        choices=("forward", "reverse", "both"),
        default="forward",
        help="Inference direction to run. Use 'both' to emit forward and reverse outputs.",
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "default", "fp16", "bf16"),
        default="auto",
        help=(
            "Autocast dtype for CUDA inference. "
            "'auto' reads checkpoint config.mixed_precision when available; "
            "'default' uses PyTorch's default autocast dtype."
        ),
    )
    parser.add_argument("--tta-mirror", action="store_true", help="Average mirror-based TTA over training-eligible axes.")
    parser.add_argument(
        "--tta-batch-size",
        type=int,
        default=None,
        help="Maximum number of mirror TTA variants to evaluate per forward pass. Defaults to all variants.",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=True)
    args = parser.parse_args(argv)
    if args.tta_batch_size is not None and int(args.tta_batch_size) <= 0:
        parser.error("--tta-batch-size must be a positive integer")
    if int(args.prefetch_factor) <= 0:
        parser.error("--prefetch-factor must be a positive integer")
    return args


def normalize_direction_value(direction: str) -> str:
    return str(direction).strip().lower()


def resolve_run_directions(direction: str) -> tuple[str, ...]:
    normalized = str(direction).strip().lower()
    if normalized == "both":
        return ("forward", "reverse")
    return (normalized,)


def append_direction_suffix(path: Path, direction: str) -> Path:
    return path.with_name(f"{path.stem}_{direction}{path.suffix}")


def resolve_single_output_path(output_tiff: Path, *, direction: str, requested_direction: str) -> Path:
    if str(requested_direction) != "both":
        return output_tiff
    if str(direction) == "forward":
        return output_tiff
    return append_direction_suffix(output_tiff, str(direction))


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s] %(levelname)s %(message)s",
    )


def is_url_like_path(path: str | Path) -> bool:
    parsed = urlparse(str(path))
    return bool(parsed.scheme and parsed.netloc)


def normalize_input_zarr_arg(path: str | Path | None) -> str | Path | None:
    if path is None:
        return None
    path_str = str(path)
    if is_url_like_path(path_str):
        return path_str
    return Path(path_str)


def open_zarr_readonly(path: str | Path):
    path_str = str(path)
    if is_url_like_path(path_str):
        import fsspec

        return zarr.open(fsspec.get_mapper(path_str.rstrip("/")), mode="r")
    return zarr.open(path_str, mode="r")


def load_grayscale_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    image = tifffile.imread(str(path))
    image = np.asarray(image)
    image = np.squeeze(image)
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Mask must be 2D, got shape={tuple(image.shape)} from {path}")
    target_h, target_w = int(target_shape[0]), int(target_shape[1])
    out = np.zeros((target_h, target_w), dtype=bool)
    h = min(target_h, int(image.shape[0]))
    w = min(target_w, int(image.shape[1]))
    out[:h, :w] = image[:h, :w] != 0
    if tuple(image.shape[:2]) != (target_h, target_w):
        LOGGER.warning(
            "Mask shape %s did not match zarr shape %s. Applied top-left crop/pad alignment.",
            tuple(int(v) for v in image.shape[:2]),
            (target_h, target_w),
        )
    return out


def downsample_mask_any(mask: np.ndarray, scale_y: int, scale_x: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if scale_y == 1 and scale_x == 1:
        return mask

    pad_h = (-mask.shape[0]) % scale_y
    pad_w = (-mask.shape[1]) % scale_x
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
    out_h = mask.shape[0] // scale_y
    out_w = mask.shape[1] // scale_x
    mask = mask.reshape(out_h, scale_y, out_w, scale_x)
    return mask.any(axis=(1, 3))


def _sliding_positions_1d(length: int, patch_size: int, stride: int) -> list[int]:
    length = int(length)
    patch_size = int(patch_size)
    stride = max(1, int(stride))
    if length <= patch_size:
        return [0]
    positions = list(range(0, max(1, length - patch_size + 1), stride))
    last = max(0, length - patch_size)
    if positions[-1] != last:
        positions.append(last)
    return positions


def iter_blocks(
    image_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    mask_lowres: np.ndarray | None,
    occupancy_scale: tuple[int, int],
) -> list[Block]:
    image_h, image_w = [int(v) for v in image_shape]
    patch_h = int(patch_size)
    patch_w = int(patch_size)
    scale_y, scale_x = [max(1, int(v)) for v in occupancy_scale]
    blocks: list[Block] = []
    y_positions = _sliding_positions_1d(image_h, patch_h, stride)
    x_positions = _sliding_positions_1d(image_w, patch_w, stride)

    for y0 in y_positions:
        valid_h = min(patch_h, image_h - y0)
        for x0 in x_positions:
            valid_w = min(patch_w, image_w - x0)
            if mask_lowres is not None:
                low_y0 = y0 // scale_y
                low_x0 = x0 // scale_x
                low_y1 = max(low_y0 + 1, math.ceil((y0 + valid_h) / scale_y))
                low_x1 = max(low_x0 + 1, math.ceil((x0 + valid_w) / scale_x))
                if not mask_lowres[low_y0:low_y1, low_x0:low_x1].any():
                    continue
            blocks.append(Block(y0=y0, x0=x0, valid_h=valid_h, valid_w=valid_w))
    return blocks


def flip_tensor_for_patch_axes(tensor: torch.Tensor, patch_axes: Sequence[int]) -> torch.Tensor:
    axes = tuple(int(axis) for axis in patch_axes)
    if not axes:
        return tensor
    return torch.flip(tensor, dims=[axis + 2 for axis in axes])


def unflip_spatial_output_for_patch_axes(output: torch.Tensor, patch_axes: Sequence[int]) -> torch.Tensor:
    spatial_dims = []
    for axis in patch_axes:
        axis = int(axis)
        if axis == 1:
            spatial_dims.append(2)
        elif axis == 2:
            spatial_dims.append(3)
    if not spatial_dims:
        return output
    return torch.flip(output, dims=spatial_dims)


def logits_to_probabilities(
    logits: torch.Tensor,
    *,
    image_hw: tuple[int, int],
    logged_resize: bool,
) -> tuple[torch.Tensor, bool]:
    if logits.ndim != 4:
        raise ValueError(
            f"Expected model logits shaped [B, C, H, W], got {tuple(logits.shape)}"
        )
    probs_t = torch.sigmoid(logits).to(dtype=torch.float32)
    if probs_t.shape[-2:] != tuple(int(v) for v in image_hw):
        if not logged_resize:
            LOGGER.info(
                "Resizing model output from %s to input patch size %s for stitching.",
                tuple(int(v) for v in probs_t.shape[-2:]),
                tuple(int(v) for v in image_hw),
            )
            logged_resize = True
        probs_t = F.interpolate(
            probs_t,
            size=image_hw,
            mode="bilinear",
            align_corners=False,
        )
    return probs_t, logged_resize


def predict_with_mirror_tta(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    tta_axes: Sequence[int],
    tta_batch_size: int | None,
    logged_resize: bool,
) -> tuple[torch.Tensor, bool]:
    variants = iter_mirror_axes(tta_axes)
    if len(variants) == 1:
        logits = model(images)
        return logits_to_probabilities(
            logits,
            image_hw=tuple(int(v) for v in images.shape[-2:]),
            logged_resize=logged_resize,
        )

    batch_size = int(images.shape[0])
    max_variants_per_forward = len(variants) if tta_batch_size is None else min(int(tta_batch_size), len(variants))
    probability_sum = None
    for start in range(0, len(variants), max_variants_per_forward):
        chunk_variants = variants[start:start + max_variants_per_forward]
        batched_variant_images = torch.cat(
            [flip_tensor_for_patch_axes(images, patch_axes) for patch_axes in chunk_variants],
            dim=0,
        )
        logits = model(batched_variant_images)
        probs_t, logged_resize = logits_to_probabilities(
            logits,
            image_hw=tuple(int(v) for v in images.shape[-2:]),
            logged_resize=logged_resize,
        )
        variant_probabilities = probs_t.reshape(
            len(chunk_variants),
            batch_size,
            *probs_t.shape[1:],
        )
        for variant_index, patch_axes in enumerate(chunk_variants):
            variant_probs = unflip_spatial_output_for_patch_axes(
                variant_probabilities[variant_index],
                patch_axes,
            )
            probability_sum = (
                variant_probs if probability_sum is None else probability_sum + variant_probs
            )
    assert probability_sum is not None
    return probability_sum / float(len(variants)), logged_resize


class OmeZarrPatchReader:
    def __init__(
        self,
        *,
        input_path: str | Path,
        resolution: str,
        depth_axis_first: bool,
        height: int,
        width: int,
        layer_indices: np.ndarray,
        preprocessing: str = "legacy_uint8",
    ):
        self.input_path = input_path if is_url_like_path(input_path) else Path(input_path)
        self.resolution = str(resolution)
        self.depth_axis_first = bool(depth_axis_first)
        self.height = int(height)
        self.width = int(width)
        self.layer_indices = np.asarray(layer_indices, dtype=np.int64)
        self.preprocessing = str(preprocessing)
        self._array = None
        self._read_mode = "fancy"
        self._z_start = int(self.layer_indices[0])
        self._z_stop = int(self.layer_indices[-1]) + 1
        if self.layer_indices.size > 1 and np.all(np.diff(self.layer_indices) == 1):
            self._read_mode = "slice_asc"
        elif self.layer_indices.size > 1 and np.all(np.diff(self.layer_indices) == -1):
            self._read_mode = "slice_desc"
            self._z_start = int(self.layer_indices[-1])
            self._z_stop = int(self.layer_indices[0]) + 1

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None
        return state

    def _ensure_array(self):
        if self._array is None:
            root = open_zarr_readonly(self.input_path)
            self._array = root if isinstance(root, zarr.Array) else root[self.resolution]
        return self._array

    def _read_raw(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        arr = self._ensure_array()
        if self.depth_axis_first:
            if self._read_mode == "slice_asc":
                block = arr[self._z_start:self._z_stop, y0:y1, x0:x1]
            elif self._read_mode == "slice_desc":
                block = arr[self._z_start:self._z_stop, y0:y1, x0:x1][::-1]
            else:
                block = arr[self.layer_indices, y0:y1, x0:x1]
            block = np.asarray(block)
            return np.transpose(block, (1, 2, 0))

        if self._read_mode == "slice_asc":
            block = arr[y0:y1, x0:x1, self._z_start:self._z_stop]
        elif self._read_mode == "slice_desc":
            block = arr[y0:y1, x0:x1, self._z_start:self._z_stop][..., ::-1]
        else:
            block = arr[y0:y1, x0:x1, self.layer_indices]
        return np.asarray(block)

    def read(self, y0: int, x0: int, out_h: int, out_w: int) -> np.ndarray:
        y0 = int(y0)
        x0 = int(x0)
        out_h = int(out_h)
        out_w = int(out_w)
        y1 = y0 + out_h
        x1 = x0 + out_w
        yy0 = max(0, y0)
        xx0 = max(0, x0)
        yy1 = min(self.height, y1)
        xx1 = min(self.width, x1)

        if self.preprocessing == "tifxyz_robust":
            out = np.zeros((out_h, out_w, self.layer_indices.size), dtype=np.float32)
        else:
            out = np.zeros((out_h, out_w, self.layer_indices.size), dtype=np.uint8)
        if yy1 <= yy0 or xx1 <= xx0:
            return out

        block = self._read_raw(yy0, yy1, xx0, xx1)
        if self.preprocessing == "tifxyz_robust":
            block = np.asarray(block, dtype=np.float32)
        else:
            block = convert_volume_dtype(block)
            np.clip(block, 0, 200, out=block)
        out[yy0 - y0:yy1 - y0, xx0 - x0:xx1 - x0] = block
        return out


def convert_volume_dtype(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data)
    if data.dtype == np.uint8:
        return np.ascontiguousarray(data)
    if data.dtype == np.uint16:
        return np.ascontiguousarray((data >> 8).astype(np.uint8, copy=False))
    if np.issubdtype(data.dtype, np.integer):
        return np.ascontiguousarray(np.clip(data, 0, 255).astype(np.uint8, copy=False))
    if np.issubdtype(data.dtype, np.floating):
        return np.ascontiguousarray(np.clip(data, 0, 255).astype(np.uint8, copy=False))
    raise TypeError(f"Unsupported zarr dtype {data.dtype}")


def choose_pyramid_array(
    root: zarr.Array | zarr.Group,
    *,
    preferred_key: str,
    purpose: str,
) -> tuple[str, zarr.Array]:
    if isinstance(root, zarr.Array):
        return "0", root

    available_keys = [str(key) for key in root.array_keys()]
    if not available_keys:
        raise ValueError(f"No arrays found in zarr group for {purpose}.")
    if preferred_key in available_keys:
        return preferred_key, root[preferred_key]

    numeric_keys = [key for key in available_keys if key.isdigit()]
    if numeric_keys and preferred_key.isdigit():
        preferred_level = int(preferred_key)
        chosen_key = min(
            numeric_keys,
            key=lambda value: (abs(int(value) - preferred_level), int(value)),
        )
    else:
        chosen_key = sorted(available_keys)[0]
    LOGGER.warning(
        "Requested %s level %s was not found. Using level %s instead.",
        purpose,
        preferred_key,
        chosen_key,
    )
    return chosen_key, root[chosen_key]


def compute_nonempty_mask_from_lowres_array(array: zarr.Array) -> tuple[np.ndarray, bool]:
    array_shape = tuple(int(v) for v in array.shape)
    if len(array_shape) == 2:
        return np.asarray(array[:]) != 0, False
    if len(array_shape) != 3:
        raise ValueError(f"Expected a 2D or 3D occupancy array, got shape={array_shape}")
    depth_axis_first = int(np.argmin(array_shape)) == 0
    lowres = np.asarray(array[:])
    if depth_axis_first:
        return np.any(lowres != 0, axis=0), True
    return np.any(lowres != 0, axis=2), False


def build_lowres_block_mask(
    root: zarr.Array | zarr.Group,
    *,
    height: int,
    width: int,
    user_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, tuple[int, int], str | None]:
    if isinstance(root, zarr.Array):
        lowres_mask = None
        occupancy_level = None
        occupancy_h = height
        occupancy_w = width
    else:
        occupancy_level, occupancy_arr = choose_pyramid_array(
            root,
            preferred_key=DEFAULT_OCCUPANCY_SCAN_LEVEL,
            purpose="occupancy scan",
        )
        lowres_mask, _ = compute_nonempty_mask_from_lowres_array(occupancy_arr)
        occupancy_h, occupancy_w = [int(v) for v in lowres_mask.shape]

    scale_y = max(1, int(round(height / max(1, occupancy_h))))
    scale_x = max(1, int(round(width / max(1, occupancy_w))))
    if user_mask is not None:
        mask_lowres = downsample_mask_any(user_mask, scale_y, scale_x)
        lowres_mask = mask_lowres if lowres_mask is None else np.logical_and(lowres_mask, mask_lowres)
    return lowres_mask, (scale_y, scale_x), occupancy_level


def iter_overlapping_chunks(
    y0: int,
    x0: int,
    valid_h: int,
    valid_w: int,
    chunk_shape: tuple[int, int],
) -> Iterator[ChunkKey]:
    chunk_h, chunk_w = [max(1, int(v)) for v in chunk_shape]
    y1 = y0 + max(1, int(valid_h)) - 1
    x1 = x0 + max(1, int(valid_w)) - 1
    for chunk_row in range(int(y0) // chunk_h, y1 // chunk_h + 1):
        for chunk_col in range(int(x0) // chunk_w, x1 // chunk_w + 1):
            yield ChunkKey(chunk_row, chunk_col)


def compute_chunk_contribution_counts(
    blocks: Sequence[Block],
    *,
    chunk_shape: tuple[int, int],
) -> dict[ChunkKey, int]:
    counts: dict[ChunkKey, int] = {}
    for block in blocks:
        for chunk_key in iter_overlapping_chunks(
            block.y0,
            block.x0,
            block.valid_h,
            block.valid_w,
            chunk_shape,
        ):
            counts[chunk_key] = counts.get(chunk_key, 0) + 1
    return counts


class ChunkAccumulator:
    def __init__(
        self,
        *,
        shape: tuple[int, int],
        chunk_shape: tuple[int, int],
        prob_sum_store,
        weight_sum_store,
        contribution_counts: dict[ChunkKey, int],
    ):
        self.height, self.width = [int(v) for v in shape]
        self.chunk_h, self.chunk_w = [max(1, int(v)) for v in chunk_shape]
        self.prob_sum_store = prob_sum_store
        self.weight_sum_store = weight_sum_store
        self.contribution_counts = dict(contribution_counts)
        self.seen_counts: dict[ChunkKey, int] = {}
        self.buffers: dict[ChunkKey, tuple[np.ndarray, np.ndarray]] = {}

    def _chunk_bounds(self, chunk_key: ChunkKey) -> tuple[int, int, int, int]:
        y0 = chunk_key.row * self.chunk_h
        x0 = chunk_key.col * self.chunk_w
        y1 = min(self.height, y0 + self.chunk_h)
        x1 = min(self.width, x0 + self.chunk_w)
        return y0, y1, x0, x1

    def _get_buffers(self, chunk_key: ChunkKey) -> tuple[np.ndarray, np.ndarray]:
        buffers = self.buffers.get(chunk_key)
        if buffers is not None:
            return buffers
        y0, y1, x0, x1 = self._chunk_bounds(chunk_key)
        buffers = (
            np.zeros((y1 - y0, x1 - x0), dtype=np.float32),
            np.zeros((y1 - y0, x1 - x0), dtype=np.float32),
        )
        self.buffers[chunk_key] = buffers
        return buffers

    def add_tile(self, *, y0: int, x0: int, tile: np.ndarray, tile_weights: np.ndarray) -> None:
        valid_h, valid_w = [int(v) for v in tile.shape]
        y1 = y0 + valid_h
        x1 = x0 + valid_w
        for chunk_key in iter_overlapping_chunks(y0, x0, valid_h, valid_w, (self.chunk_h, self.chunk_w)):
            chunk_y0, chunk_y1, chunk_x0, chunk_x1 = self._chunk_bounds(chunk_key)
            intersect_y0 = max(y0, chunk_y0)
            intersect_y1 = min(y1, chunk_y1)
            intersect_x0 = max(x0, chunk_x0)
            intersect_x1 = min(x1, chunk_x1)
            if intersect_y1 <= intersect_y0 or intersect_x1 <= intersect_x0:
                continue
            prob_buffer, weight_buffer = self._get_buffers(chunk_key)
            tile_y0 = intersect_y0 - y0
            tile_y1 = intersect_y1 - y0
            tile_x0 = intersect_x0 - x0
            tile_x1 = intersect_x1 - x0
            chunk_local_y0 = intersect_y0 - chunk_y0
            chunk_local_y1 = intersect_y1 - chunk_y0
            chunk_local_x0 = intersect_x0 - chunk_x0
            chunk_local_x1 = intersect_x1 - chunk_x0
            tile_weights_view = tile_weights[tile_y0:tile_y1, tile_x0:tile_x1]
            prob_buffer[chunk_local_y0:chunk_local_y1, chunk_local_x0:chunk_local_x1] += (
                tile[tile_y0:tile_y1, tile_x0:tile_x1] * tile_weights_view
            )
            weight_buffer[chunk_local_y0:chunk_local_y1, chunk_local_x0:chunk_local_x1] += tile_weights_view

            seen_count = self.seen_counts.get(chunk_key, 0) + 1
            total_count = self.contribution_counts[chunk_key]
            if seen_count >= total_count:
                self._flush_chunk(chunk_key)
            else:
                self.seen_counts[chunk_key] = seen_count

    def _flush_chunk(self, chunk_key: ChunkKey) -> None:
        prob_buffer, weight_buffer = self.buffers.pop(chunk_key)
        self.seen_counts.pop(chunk_key, None)
        y0, y1, x0, x1 = self._chunk_bounds(chunk_key)
        self.prob_sum_store[y0:y1, x0:x1] = prob_buffer
        self.weight_sum_store[y0:y1, x0:x1] = weight_buffer

    def flush_remaining(self) -> None:
        for chunk_key in list(self.buffers.keys()):
            self._flush_chunk(chunk_key)


class OmeZarrBlockDataset(Dataset):
    def __init__(
        self,
        *,
        reader: OmeZarrPatchReader,
        blocks: Sequence[Block],
        patch_size: int,
        preprocessing: str = "legacy_uint8",
    ):
        self.reader = reader
        self.blocks = list(blocks)
        self.patch_h = int(patch_size)
        self.patch_w = int(patch_size)
        self.preprocessing = str(preprocessing)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int):
        block = self.blocks[index]
        patch = self.reader.read(block.y0, block.x0, self.patch_h, self.patch_w)
        patch = np.moveaxis(patch, -1, 0)
        if self.preprocessing == "tifxyz_robust":
            from vesuvius.image_proc.intensity.normalization import normalize_robust

            patch = normalize_robust(patch)
        else:
            patch = patch.astype(np.float32, copy=False) / 255.0
        # Models trained in this repo consume channel-first 3D volumes: [C, Z, Y, X].
        image = torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)
        meta = torch.tensor([block.y0, block.x0, block.valid_h, block.valid_w], dtype=torch.int64)
        return image, meta


def load_checkpoint_payload(checkpoint_path: Path | str) -> Any:
    try:
        return torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(checkpoint_path), map_location="cpu")


def extract_state_dict_from_payload(payload: Any, checkpoint_path: Path | str) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and isinstance(payload.get("model_state_dict"), dict):
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state_dict = payload["model"]
    elif isinstance(payload, dict) and all(isinstance(v, torch.Tensor) for v in payload.values()):
        state_dict = payload
    else:
        raise ValueError(f"Unsupported checkpoint format for checkpoint={str(checkpoint_path)!r}")

    if state_dict and all(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def pretrained_backbone_looks_like_checkpoint(value: Any) -> bool:
    if not isinstance(value, (str, Path)):
        return False
    text = str(value).strip()
    if not text:
        return False
    if text.startswith(("~", ".", "/")):
        return True
    if "/" in text or "\\" in text:
        return True
    return Path(text).suffix.lower() in {".pt", ".pth"}


def resolve_checkpoint_reference_path(
    reference: str | Path,
    *,
    relative_to: Path | str | None,
) -> Path:
    reference_path = Path(str(reference)).expanduser()
    candidates: list[Path] = []
    if reference_path.is_absolute():
        candidates.append(reference_path)
    else:
        if relative_to is not None:
            base_path = Path(str(relative_to)).expanduser()
            if base_path.suffix:
                base_path = base_path.parent
            candidates.append(base_path / reference_path)
        candidates.append(reference_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve(strict=False)


def resolve_tifxyz_pretrained_backbone_config(
    config: dict[str, Any],
    *,
    checkpoint_path: Path | str,
    seen_paths: set[Path] | None = None,
) -> dict[str, Any]:
    model_config = config.get("model_config")
    if not isinstance(model_config, dict):
        return config

    pretrained_backbone = model_config.get("pretrained_backbone")
    if not pretrained_backbone_looks_like_checkpoint(pretrained_backbone):
        return config

    resolved_backbone_ckpt = resolve_checkpoint_reference_path(
        pretrained_backbone,
        relative_to=checkpoint_path,
    )
    if seen_paths is None:
        seen_paths = set()
    if resolved_backbone_ckpt in seen_paths:
        chain = " -> ".join(str(path) for path in [*seen_paths, resolved_backbone_ckpt])
        raise ValueError(f"Detected recursive pretrained_backbone checkpoint chain: {chain}")
    if not resolved_backbone_ckpt.exists():
        raise FileNotFoundError(
            f"Resolved pretrained_backbone checkpoint does not exist: {resolved_backbone_ckpt}"
        )

    backbone_payload = load_checkpoint_payload(resolved_backbone_ckpt)
    backbone_config = backbone_payload.get("config") if isinstance(backbone_payload, dict) else None
    if not isinstance(backbone_config, dict):
        raise ValueError(
            "Checkpoint-backed pretrained_backbone must point to a training checkpoint "
            f"with a dict config, got {resolved_backbone_ckpt}"
        )

    nested_seen_paths = set(seen_paths)
    nested_seen_paths.add(resolved_backbone_ckpt)
    resolved_backbone_config = resolve_tifxyz_pretrained_backbone_config(
        copy.deepcopy(backbone_config),
        checkpoint_path=resolved_backbone_ckpt,
        seen_paths=nested_seen_paths,
    )

    resolved_model_config = resolved_backbone_config.get("model_config")
    resolved_pretrained_backbone = (
        resolved_model_config.get("pretrained_backbone")
        if isinstance(resolved_model_config, dict)
        else None
    )
    if not resolved_pretrained_backbone:
        raise ValueError(
            "Checkpoint-backed pretrained_backbone must ultimately resolve to a concrete backbone name, "
            f"but {resolved_backbone_ckpt} did not define model_config.pretrained_backbone"
        )

    if pretrained_backbone_looks_like_checkpoint(resolved_pretrained_backbone):
        raise ValueError(
            "Checkpoint-backed pretrained_backbone resolution did not terminate in a concrete backbone name: "
            f"{resolved_pretrained_backbone!r}"
        )

    updated_config = copy.deepcopy(config)
    updated_model_config = dict(updated_config.get("model_config") or {})
    updated_model_config["pretrained_backbone"] = resolved_pretrained_backbone
    updated_config["model_config"] = updated_model_config
    LOGGER.info(
        "Resolved checkpoint-backed pretrained_backbone %r via %s -> %r",
        pretrained_backbone,
        resolved_backbone_ckpt,
        resolved_pretrained_backbone,
    )
    return updated_config


def checkpoint_looks_like_tifxyz_training(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    config = payload.get("config")
    model_state = payload.get("model")
    if not isinstance(config, dict) or not isinstance(model_state, dict):
        return False
    model_type = str(config.get("model_type", "")).strip().lower()
    patch_size = config.get("patch_size")
    return model_type == "unet" and isinstance(patch_size, (list, tuple)) and len(patch_size) == 3


def checkpoint_amp_dtype(payload: Any, checkpoint_path: Path | str) -> torch.dtype | None:
    if not isinstance(payload, dict):
        return None
    config = payload.get("config")
    if not isinstance(config, dict):
        return None

    mixed_precision = config.get("mixed_precision")
    if mixed_precision is None:
        LOGGER.info(
            "Checkpoint %s config does not define mixed_precision; using default CUDA autocast dtype.",
            checkpoint_path,
        )
        return None

    mixed_precision = str(mixed_precision).strip().lower()
    if mixed_precision in {"fp16", "float16", "half"}:
        LOGGER.info(
            "Checkpoint %s requested AMP dtype=float16 from config.mixed_precision=%r.",
            checkpoint_path,
            mixed_precision,
        )
        return torch.float16
    if mixed_precision in {"bf16", "bfloat16"}:
        LOGGER.info(
            "Checkpoint %s requested AMP dtype=bfloat16 from config.mixed_precision=%r.",
            checkpoint_path,
            mixed_precision,
        )
        return torch.bfloat16
    if mixed_precision in {"no", "none", "false", "off", "disabled"}:
        LOGGER.info(
            "Checkpoint %s config.mixed_precision=%r does not specify an AMP dtype; using default CUDA autocast dtype.",
            checkpoint_path,
            mixed_precision,
        )
        return None

    LOGGER.warning(
        "Checkpoint %s has unsupported config.mixed_precision=%r; using default CUDA autocast dtype.",
        checkpoint_path,
        mixed_precision,
    )
    return None


def resolve_amp_dtype(
    *,
    amp_dtype_arg: str,
    checkpoint_payload: Any,
    checkpoint_path: Path | str,
) -> torch.dtype | None:
    amp_dtype_arg = str(amp_dtype_arg).strip().lower()
    if amp_dtype_arg == "default":
        return None
    if amp_dtype_arg == "fp16":
        return torch.float16
    if amp_dtype_arg == "bf16":
        return torch.bfloat16
    if amp_dtype_arg != "auto":
        raise ValueError(f"Unsupported --amp-dtype value: {amp_dtype_arg!r}")
    return checkpoint_amp_dtype(checkpoint_payload, checkpoint_path)


def tifxyz_roi_size_from_config(config: dict[str, Any]) -> int:
    patch_size = config.get("crop_size", config.get("patch_size"))
    if not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3:
        raise ValueError(
            "tifxyz checkpoint config must define crop_size or patch_size as [z, y, x], "
            f"got {patch_size!r}"
        )
    roi_y = int(patch_size[1])
    roi_x = int(patch_size[2])
    if roi_y != roi_x:
        raise ValueError(
            "inference_ome_zarr.py currently requires square tifxyz patches for inference, "
            f"got patch_size={tuple(int(v) for v in patch_size)}"
        )
    return roi_y


def tifxyz_in_chans_from_config(config: dict[str, Any]) -> int:
    patch_size = config.get("crop_size", config.get("patch_size"))
    if isinstance(patch_size, (list, tuple)) and len(patch_size) == 3:
        return int(patch_size[0])
    raise ValueError(
        "tifxyz checkpoint config must define crop_size or patch_size as [z, y, x] "
        f"to determine inference depth, got {patch_size!r}"
    )


def build_tifxyz_model_bundle(payload: dict[str, Any], checkpoint_path: Path | str) -> ConfiguredModel:
    from vesuvius.neural_tracing.nets.models import make_model

    config = copy.deepcopy(payload["config"])
    config = resolve_tifxyz_pretrained_backbone_config(config, checkpoint_path=checkpoint_path)
    model_type = str(config.get("model_type", "")).strip().lower()
    if model_type != "unet":
        raise ValueError(
            "inference_ome_zarr.py only supports tifxyz checkpoints trained with model_type='unet', "
            f"got {model_type!r}"
        )
    config.setdefault("crop_size", config.get("patch_size"))
    targets_cfg = config.setdefault("targets", {})
    ink_cfg = targets_cfg.setdefault("ink", {})
    ink_cfg["out_channels"] = 1
    ink_cfg.setdefault("activation", "none")

    base_model = make_model(config)
    state_dict = extract_state_dict_from_payload(payload, checkpoint_path)
    incompat = base_model.load_state_dict(state_dict, strict=False)
    LOGGER.info(
        "Loaded tifxyz checkpoint %s (missing_keys=%d unexpected_keys=%d)",
        checkpoint_path,
        len(incompat.missing_keys),
        len(incompat.unexpected_keys),
    )

    model = TargetHeadWrapper(base_model, target_name="ink")
    model.eval()
    return ConfiguredModel(
        model=model,
        roi_size=tifxyz_roi_size_from_config(config),
        in_chans=tifxyz_in_chans_from_config(config),
        preprocessing="tifxyz_robust",
        source="tifxyz_dataset/train.py",
        amp_dtype=None,
    )


def configure_model(args: argparse.Namespace) -> ConfiguredModel:
    checkpoint_payload = load_checkpoint_payload(args.checkpoint)
    resolved_amp_dtype = resolve_amp_dtype(
        amp_dtype_arg=args.amp_dtype,
        checkpoint_payload=checkpoint_payload,
        checkpoint_path=args.checkpoint,
    )
    explicit_tifxyz = args.model_type == "tifxyz_unet"
    auto_tifxyz = args.model_type in {"auto", "residual_unet"} and checkpoint_looks_like_tifxyz_training(checkpoint_payload)
    if explicit_tifxyz or auto_tifxyz:
        if args.metadata_json is not None:
            LOGGER.info("Ignoring --metadata-json for tifxyz checkpoint %s", args.checkpoint)
        configured_model = build_tifxyz_model_bundle(checkpoint_payload, args.checkpoint)
        configured_model.amp_dtype = resolved_amp_dtype
        return configured_model
    if args.model_type == "auto":
        raise ValueError(
            "Could not infer a supported model type from the checkpoint. "
            "Pass --model-type resnet3d, residual_unet, or tifxyz_unet explicitly."
        )

    from train_resnet3d_lib.config import CFG, apply_metadata_hyperparameters, load_metadata_json
    from train_resnet3d_lib.model import RegressionPLModel

    metadata = None
    if args.metadata_json is not None:
        metadata = load_metadata_json(str(args.metadata_json))
        apply_metadata_hyperparameters(CFG, metadata)
    elif args.model_type in {"residual_unet", "auto"}:
        LOGGER.warning(
            "No --metadata-json was provided for residual_unet. "
            "This will only work if the default CFG matches the checkpoint architecture."
        )

    CFG.init_ckpt_path = str(args.checkpoint)
    CFG.model_impl = "resnet3d_hybrid" if args.model_type == "resnet3d" else "vesuvius_resunet_hybrid"

    model = RegressionPLModel(
        size=int(getattr(CFG, "size", 256)),
        norm=str(getattr(CFG, "norm", "batch")),
        group_norm_groups=int(getattr(CFG, "group_norm_groups", 32)),
        model_impl=str(getattr(CFG, "model_impl", "resnet3d_hybrid")),
        vesuvius_model_config=getattr(CFG, "vesuvius_model_config", {}),
        vesuvius_target_name=str(getattr(CFG, "vesuvius_target_name", "ink")),
        vesuvius_z_projection_mode=str(getattr(CFG, "vesuvius_z_projection_mode", "logsumexp")),
        vesuvius_z_projection_lse_tau=float(getattr(CFG, "vesuvius_z_projection_lse_tau", 1.0)),
        vesuvius_z_projection_mlp_hidden=int(getattr(CFG, "vesuvius_z_projection_mlp_hidden", 64)),
        vesuvius_z_projection_mlp_dropout=float(getattr(CFG, "vesuvius_z_projection_mlp_dropout", 0.0)),
        vesuvius_z_projection_mlp_depth=int(
            getattr(CFG, "vesuvius_z_projection_mlp_depth", None) or getattr(CFG, "in_chans", 1)
        ),
        n_groups=1,
        group_names=["inference"],
    )

    state_dict = extract_state_dict_from_payload(checkpoint_payload, args.checkpoint)
    incompat = model.load_state_dict(state_dict, strict=False)
    LOGGER.info(
        "Loaded checkpoint %s (missing_keys=%d unexpected_keys=%d)",
        args.checkpoint,
        len(incompat.missing_keys),
        len(incompat.unexpected_keys),
    )
    model.eval()
    return ConfiguredModel(
        model=model,
        roi_size=int(getattr(CFG, "size", 256)),
        in_chans=int(getattr(CFG, "in_chans", 62)),
        preprocessing="legacy_uint8",
        source="train_resnet3d_lib",
        amp_dtype=resolved_amp_dtype,
    )


def maybe_compile_model(
    model: torch.nn.Module,
    *,
    enabled: bool,
    mode: str,
) -> torch.nn.Module:
    if not enabled:
        LOGGER.info("torch.compile disabled.")
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        LOGGER.warning("torch.compile is unavailable in this PyTorch build. Continuing without compilation.")
        return model
    try:
        compiled_model = compile_fn(model, mode=str(mode), fullgraph=False, dynamic=False)
    except Exception as exc:
        LOGGER.warning("torch.compile failed (%s). Continuing without compilation.", exc)
        return model
    LOGGER.info("Enabled torch.compile (mode=%s)", mode)
    return compiled_model


def run_block_inference(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    accumulator: ChunkAccumulator,
    weight_map: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype | None = None,
    tta_axes: Sequence[int] = (),
    tta_batch_size: int | None = None,
) -> None:
    autocast_enabled = bool(amp and device.type == "cuda")
    logged_resize = False
    tta_axes = tuple(int(axis) for axis in tta_axes)
    if autocast_enabled:
        if amp_dtype is None:
            LOGGER.info("CUDA autocast enabled for inference with default dtype.")
        else:
            LOGGER.info(
                "CUDA autocast enabled for inference with dtype=%s.",
                str(amp_dtype).replace("torch.", ""),
            )
    else:
        LOGGER.info("Autocast disabled for inference (device=%s).", device.type)
    if tta_axes:
        LOGGER.info(
            "Mirror TTA enabled over patch axes %s (%d variants, tta_batch_size=%s).",
            tta_axes,
            len(iter_mirror_axes(tta_axes)),
            "all" if tta_batch_size is None else int(tta_batch_size),
        )
    else:
        LOGGER.info("Mirror TTA disabled.")
    with torch.inference_mode():
        for images, meta in tqdm(loader, desc="Infer", unit="block"):
            images = images.to(device, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda", enabled=True, dtype=amp_dtype)
                if autocast_enabled
                else nullcontext()
            )
            with amp_context:
                if tta_axes:
                    probs_t, logged_resize = predict_with_mirror_tta(
                        model,
                        images,
                        tta_axes=tta_axes,
                        tta_batch_size=tta_batch_size,
                        logged_resize=logged_resize,
                    )
                else:
                    logits = model(images)
                    probs_t, logged_resize = logits_to_probabilities(
                        logits,
                        image_hw=tuple(int(v) for v in images.shape[-2:]),
                        logged_resize=logged_resize,
                    )
            probs = probs_t.detach().cpu().numpy()[:, 0]
            meta_np = meta.cpu().numpy()
            for i in range(probs.shape[0]):
                y0, x0, valid_h, valid_w = [int(v) for v in meta_np[i]]
                tile = np.asarray(probs[i, :valid_h, :valid_w], dtype=np.float32)
                tile_weights = np.asarray(weight_map[:valid_h, :valid_w], dtype=np.float32)
                if mask is not None:
                    mask_view = mask[y0:y0 + valid_h, x0:x0 + valid_w].astype(np.float32, copy=False)
                    tile = tile * mask_view
                    tile_weights = tile_weights * mask_view
                accumulator.add_tile(
                    y0=y0,
                    x0=x0,
                    tile=tile,
                    tile_weights=tile_weights,
                )


def iter_probability_tiles(prob_sum_store, weight_sum_store, tile_shape: tuple[int, int]) -> Iterator[np.ndarray]:
    height, width = [int(v) for v in prob_sum_store.shape]
    tile_h, tile_w = [int(v) for v in tile_shape]
    for y0 in range(0, height, tile_h):
        y1 = min(height, y0 + tile_h)
        for x0 in range(0, width, tile_w):
            x1 = min(width, x0 + tile_w)
            prob = np.asarray(prob_sum_store[y0:y1, x0:x1], dtype=np.float32)
            weights = np.asarray(weight_sum_store[y0:y1, x0:x1], dtype=np.float32)
            tile = np.divide(prob, np.clip(weights, 1e-6, None), out=np.zeros_like(prob), where=weights > 0)
            tile = np.clip(tile, 0.0, 1.0)
            yield (tile * 255.0).astype(np.uint8, copy=False)


def write_output_tiff(prob_sum_store, weight_sum_store, output_path: Path, tile_shape: tuple[int, int]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        str(output_path),
        iter_probability_tiles(prob_sum_store, weight_sum_store, tile_shape),
        shape=tuple(int(v) for v in prob_sum_store.shape),
        dtype=np.uint8,
        compression="lzw",
        tile=tuple(int(v) for v in tile_shape),
        bigtiff=True,
        metadata=None,
        software="inference_ome_zarr.py",
    )


def open_temp_zarr_array(
    path: Path,
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int],
    dtype,
):
    # zarr v2 expects "zarr_version"; older/newer variants may not accept it.
    try:
        return zarr.open(
            str(path),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
            zarr_version=2,
        )
    except TypeError:
        return zarr.open(
            str(path),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=dtype,
        )


def infer_single_zarr(
    *,
    args: argparse.Namespace,
    input_zarr: str | Path,
    configured_model: ConfiguredModel,
    output_tiff: Path,
    layer_direction: str = "forward",
) -> None:
    resolution = str(args.resolution)
    root = open_zarr_readonly(input_zarr)
    if isinstance(root, zarr.Array):
        resolution = "0"
        volume_arr = root
    else:
        volume_arr = root[resolution]

    volume_shape = tuple(int(v) for v in volume_arr.shape)
    if len(volume_shape) != 3:
        raise ValueError(f"Expected a 3D input array, got shape={volume_shape}")
    volume_chunks = tuple(int(v) for v in (volume_arr.chunks or volume_shape))
    depth_axis_first = int(np.argmin(volume_shape)) == 0
    if depth_axis_first:
        depth, height, width = volume_shape
        _, spatial_chunk_h, spatial_chunk_w = volume_chunks
    else:
        height, width, depth = volume_shape
        spatial_chunk_h, spatial_chunk_w, _ = volume_chunks

    roi_size = int(configured_model.roi_size)
    patch_size = roi_size
    patch_stride = max(1, int(round(float(patch_size) * (1.0 - float(args.overlap)))))

    tiff_tile_shape = (patch_size, patch_size)
    if tiff_tile_shape[0] % 16 != 0 or tiff_tile_shape[1] % 16 != 0:
        tiff_tile_shape = (int(spatial_chunk_h), int(spatial_chunk_w))

    in_chans = int(configured_model.in_chans)
    tta_axes = compute_equal_length_mirror_axes((in_chans, roi_size, roi_size)) if bool(args.tta_mirror) else ()
    layer_start = 0 if args.layer_start is None else int(args.layer_start)
    layer_end = depth if args.layer_end is None else int(args.layer_end)
    if layer_start < 0:
        layer_start += depth
    if layer_end < 0:
        layer_end += depth
    layer_start = max(0, layer_start)
    layer_end = min(depth, layer_end)
    if layer_end <= layer_start:
        layer_start, layer_end = 0, depth
    layer_indices = np.arange(layer_start, layer_end, dtype=np.int64)
    if layer_indices.size > in_chans:
        start_offset = (layer_indices.size - in_chans) // 2
        layer_indices = layer_indices[start_offset:start_offset + in_chans]
    if str(layer_direction) == "reverse":
        layer_indices = layer_indices[::-1]
    reader = OmeZarrPatchReader(
        input_path=input_zarr,
        resolution=resolution,
        depth_axis_first=depth_axis_first,
        height=height,
        width=width,
        layer_indices=layer_indices,
        preprocessing=configured_model.preprocessing,
    )

    layer_order = str(layer_direction)
    LOGGER.info(
        "Input level=%s shape=(depth=%d, height=%d, width=%d) chunks=(%d, %d) patch=%d stride=%d blend=%.3f tiff_tile=%s in_chans=%d layer_order=%s tta_axes=%s",
        resolution,
        depth,
        height,
        width,
        spatial_chunk_h,
        spatial_chunk_w,
        patch_size,
        patch_stride,
        float(args.overlap),
        tiff_tile_shape,
        layer_indices.size,
        layer_order,
        tta_axes,
    )

    mask = None
    if args.mask_path is not None:
        mask = load_grayscale_mask(args.mask_path, (height, width))
        LOGGER.info(
            "Loaded mask %s with foreground coverage %.3f%%",
            args.mask_path,
            100.0 * float(mask.mean()),
        )
    else:
        LOGGER.info("No mask supplied. Using the entire zarr.")

    block_mask_lowres, occupancy_scale, occupancy_level = build_lowres_block_mask(
        root,
        height=height,
        width=width,
        user_mask=mask,
    )
    if block_mask_lowres is None:
        LOGGER.warning("No low-resolution occupancy scan is available for %s. All tiles will be scheduled.", input_zarr)
    else:
        LOGGER.info(
            "Using occupancy scan level=%s shape=%s scale=(%d, %d) nonempty_coverage=%.3f%%",
            occupancy_level,
            tuple(int(v) for v in block_mask_lowres.shape),
            int(occupancy_scale[0]),
            int(occupancy_scale[1]),
            100.0 * float(block_mask_lowres.mean()),
        )

    blocks = iter_blocks(
        image_shape=(height, width),
        patch_size=patch_size,
        stride=patch_stride,
        mask_lowres=block_mask_lowres,
        occupancy_scale=occupancy_scale,
    )
    LOGGER.info("Selected %d patches for inference.", len(blocks))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OmeZarrBlockDataset(
        reader=reader,
        blocks=blocks,
        patch_size=patch_size,
        preprocessing=configured_model.preprocessing,
    )
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": int(args.batch_size),
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(**loader_kwargs)
    weight_mode = "constant" if float(args.overlap) == 0.0 else "gaussian"
    weight_map = compute_importance_map(
        patch_size=(patch_size, patch_size),
        mode=weight_mode,
        sigma_scale=0.125,
        device="cpu",
    ).cpu().numpy().astype(np.float32, copy=False)

    temp_parent = Path(tempfile.mkdtemp(prefix="ome_zarr_infer_"))
    LOGGER.info("Using temporary store %s", temp_parent)
    try:
        chunk_shape = (
            min(int(tiff_tile_shape[0]), int(height)),
            min(int(tiff_tile_shape[1]), int(width)),
        )
        contribution_counts = compute_chunk_contribution_counts(
            blocks,
            chunk_shape=chunk_shape,
        )
        prob_sum_store = open_temp_zarr_array(
            temp_parent / "prob_sum.zarr",
            shape=(height, width),
            chunks=chunk_shape,
            dtype=np.float32,
        )
        weight_sum_store = open_temp_zarr_array(
            temp_parent / "weight_sum.zarr",
            shape=(height, width),
            chunks=chunk_shape,
            dtype=np.float32,
        )
        accumulator = ChunkAccumulator(
            shape=(height, width),
            chunk_shape=chunk_shape,
            prob_sum_store=prob_sum_store,
            weight_sum_store=weight_sum_store,
            contribution_counts=contribution_counts,
        )

        if len(dataset) > 0:
            run_block_inference(
                loader=loader,
                model=configured_model.model,
                accumulator=accumulator,
                weight_map=weight_map,
                mask=mask,
                device=device,
                amp=True,
                amp_dtype=configured_model.amp_dtype,
                tta_axes=tta_axes,
                tta_batch_size=args.tta_batch_size,
            )
            accumulator.flush_remaining()
        else:
            LOGGER.warning("No occupied blocks were found. Writing an all-zero output.")

        write_output_tiff(prob_sum_store, weight_sum_store, output_tiff, tiff_tile_shape)
        LOGGER.info("Wrote %s", output_tiff)
    finally:
        shutil.rmtree(temp_parent, ignore_errors=True)


def resolve_segment_zarr_path(segment_dir: Path) -> Path:
    segment_name = segment_dir.name
    direct_candidates = [
        segment_dir / segment_name,
        segment_dir / f"{segment_name}.zarr",
        segment_dir / f"{segment_name}.ome.zarr",
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate

    discovered: list[Path] = []
    for child in sorted(segment_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.suffix == ".zarr":
            discovered.append(child)
            continue
        if (child / ".zgroup").exists() or (child / ".zarray").exists() or (child / "zarr.json").exists():
            discovered.append(child)
    if len(discovered) == 1:
        return discovered[0]

    raise FileNotFoundError(
        f"Could not find zarr for segment {segment_name!r}. Expected one of: "
        f"{', '.join(str(p) for p in direct_candidates)}"
    )


def infer_folder(args: argparse.Namespace, configured_model: ConfiguredModel) -> None:
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"--folder is not a directory: {folder}")

    checkpoint_stem = Path(args.checkpoint).stem
    run_directions = resolve_run_directions(args.direction)
    date_str = datetime.now().strftime("%d%m%y")
    output_prefix = f"{str(args.output_prefix)}_" if str(args.output_prefix) else ""

    segment_dirs = sorted(path for path in folder.iterdir() if path.is_dir())
    if not segment_dirs:
        raise FileNotFoundError(f"No subdirectories were found under --folder {folder}")

    ran_count = 0
    skipped_count = 0
    for segment_dir in segment_dirs:
        segment_name = segment_dir.name
        try:
            input_zarr = resolve_segment_zarr_path(segment_dir)
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping %s: %s", segment_dir, exc)
            skipped_count += 1
            continue

        for direction in run_directions:
            output_tiff = (
                segment_dir
                / "preds"
                / f"{output_prefix}{segment_name}_{checkpoint_stem}_{direction}_{date_str}.tif"
            )
            LOGGER.info(
                "Running segment=%s input=%s output=%s direction=%s",
                segment_name,
                input_zarr,
                output_tiff,
                direction,
            )
            infer_single_zarr(
                args=args,
                configured_model=configured_model,
                input_zarr=input_zarr,
                output_tiff=output_tiff,
                layer_direction=direction,
            )
            ran_count += 1

    LOGGER.info("Folder run complete. segments_ran=%d segments_skipped=%d", ran_count, skipped_count)


def normalize_inference_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.input_zarr = normalize_input_zarr_arg(args.input_zarr)
    args.output_tiff = Path(args.output_tiff) if args.output_tiff is not None else None
    args.folder = Path(args.folder) if args.folder is not None else None
    args.checkpoint = Path(args.checkpoint) if args.checkpoint is not None else None
    args.checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else None

    if args.checkpoint_path is not None:
        args.checkpoint = args.checkpoint_path

    args.direction = normalize_direction_value(args.direction)

    # Convenience: allow "--folder <dir> <checkpoint>" without --checkpoint-path.
    if args.folder is not None and args.checkpoint is None and args.input_zarr is not None and args.output_tiff is None:
        args.checkpoint = Path(str(args.input_zarr))
        args.input_zarr = None

    if args.checkpoint is None:
        raise ValueError("Checkpoint is required. Provide positional <checkpoint> or --checkpoint-path.")

    if args.folder is not None:
        if args.output_tiff is not None:
            raise ValueError("Do not pass output_tiff positional argument with --folder.")
    else:
        if args.input_zarr is None or args.output_tiff is None:
            raise ValueError(
                "Single-zarr mode requires positional: <input_zarr> <checkpoint> <output_tiff>."
            )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = normalize_inference_paths(parse_args(argv))
    configure_logging("INFO")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    configured_model = configure_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_model.model = configured_model.model.to(device)
    configured_model.model = maybe_compile_model(
        configured_model.model,
        enabled=bool(args.compile_model),
        mode=str(args.compile_mode),
    )
    LOGGER.info(
        "Configured model source=%s roi_size=%d in_chans=%d preprocessing=%s compile=%s",
        configured_model.source,
        configured_model.roi_size,
        configured_model.in_chans,
        configured_model.preprocessing,
        bool(args.compile_model),
    )
    if args.folder is not None:
        infer_folder(args=args, configured_model=configured_model)
    else:
        for direction in resolve_run_directions(args.direction):
            output_tiff = resolve_single_output_path(
                args.output_tiff,
                direction=direction,
                requested_direction=args.direction,
            )
            LOGGER.info("Running input=%s output=%s direction=%s", args.input_zarr, output_tiff, direction)
            infer_single_zarr(
                args=args,
                configured_model=configured_model,
                input_zarr=args.input_zarr,
                output_tiff=output_tiff,
                layer_direction=direction,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
