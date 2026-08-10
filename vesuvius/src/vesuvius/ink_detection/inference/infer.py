"""Checkpoint-driven flat surface-volume inference to uint8 TIFF."""

from __future__ import annotations

import itertools
import logging
import math
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from urllib.parse import urlparse

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import zarr

from vesuvius.ink_detection.models.checkpoint import (
    load_checkpoint,
    resolve_pretrained_backbone_config,
    select_inference_weights,
)
from vesuvius.ink_detection.config import InkConfig, NormalizationConfig
from vesuvius.ink_detection.inference.inference_runtime import (
    parse_gpu_ids,
    prepare_model_for_inference,
    resolve_amp_dtype,
)
from vesuvius.ink_detection.models.input_padding import center_pad_input_depth
from vesuvius.ink_detection.models.model import make_model
from vesuvius.ink_detection.volume_io import (
    open_volume,
    open_volume_root,
    select_volume_level,
)
from vesuvius.utils.cli import HyphenUnderscoreParser


LOGGER = logging.getLogger(__name__)
DEFAULT_OCCUPANCY_SCAN_LEVEL = "3"
DEFAULT_OVERLAP = 0.25
DEFAULT_PREFETCH_FACTOR = 2


@dataclass(frozen=True)
class Block:
    """One top-left flat patch and its in-volume spatial extent."""

    y0: int
    x0: int
    valid_h: int
    valid_w: int


@dataclass(frozen=True)
class ChunkKey:
    """A row/column output chunk identity."""

    row: int
    col: int


@dataclass(frozen=True)
class ConfiguredModel:
    """The checkpoint-derived flat model and preprocessing contract."""

    model: nn.Module
    patch_size: int
    input_depth: int
    preprocessing: str
    amp_dtype: torch.dtype | None
    config: InkConfig
    selected_state: str


class FlatTargetModel(nn.Module):
    """Project the configured ink target to one BCYX logits tensor."""

    def __init__(self, model: nn.Module, *, input_pad_depth_to: int | None) -> None:
        super().__init__()
        self.model = model
        self.input_pad_depth_to = input_pad_depth_to

    def forward(self, image_BCZYX: torch.Tensor) -> torch.Tensor:
        image_BCZYX = center_pad_input_depth(
            image_BCZYX, self.input_pad_depth_to
        )
        outputs = self.model(image_BCZYX)
        if not isinstance(outputs, Mapping):
            raise TypeError(
                "Ink model must return a target mapping, got "
                f"{type(outputs).__name__}"
            )
        if "ink" not in outputs:
            raise KeyError(
                f"Ink model output is missing 'ink'; available: {sorted(outputs)}"
            )
        logits = outputs["ink"]
        if isinstance(logits, (list, tuple)):
            if not logits:
                raise ValueError("Ink model returned an empty deep-supervision output")
            logits = logits[0]
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"Ink logits must be a tensor, got {type(logits).__name__}"
            )
        return logits


def flat_preprocessing_from_config(config: NormalizationConfig) -> str:
    """Map training normalization settings to flat-inference preprocessing."""

    if config.mode == "divide":
        if config.divisor != 255.0:
            raise ValueError(
                "Flat inference divide preprocessing requires divisor=255, "
                f"got {config.divisor!r}"
            )
        return "divide_255"
    if config.mode == "clip_divide":
        parameters = (config.clip_min, config.clip_max, config.divisor)
        if parameters != (0.0, 200.0, 255.0):
            raise ValueError(
                "Flat inference supports clip_divide only with clip_min=0, "
                "clip_max=200, divisor=255"
            )
        return "legacy_uint8"
    return "tifxyz_robust"


def convert_volume_dtype(data: np.ndarray) -> np.ndarray:
    """Convert legacy flat inference input to contiguous uint8 values."""

    data = np.asarray(data)
    if data.dtype == np.uint8:
        return np.ascontiguousarray(data)
    if data.dtype == np.uint16:
        return np.ascontiguousarray((data >> 8).astype(np.uint8, copy=False))
    if np.issubdtype(data.dtype, np.integer) or np.issubdtype(
        data.dtype, np.floating
    ):
        clipped = np.clip(data, 0, 255)
        return np.ascontiguousarray(clipped.astype(np.uint8, copy=False))
    raise TypeError(f"Unsupported surface-volume dtype {data.dtype}")


def normalize_flat_patch(
    patch_ZYX: np.ndarray,
    preprocessing: str,
) -> np.ndarray:
    """Normalize one reader-processed ZYX patch by the legacy flat contract."""

    if preprocessing == "tifxyz_robust":
        from vesuvius.image_proc.intensity.normalization import normalize_robust

        return np.ascontiguousarray(normalize_robust(patch_ZYX))
    if preprocessing in {"legacy_uint8", "divide_255"}:
        normalized = np.ascontiguousarray(patch_ZYX, dtype=np.float32)
        normalized *= 1.0 / 255.0
        return normalized
    raise ValueError(f"Unsupported flat preprocessing {preprocessing!r}")


def compute_equal_length_mirror_axes(shape: Sequence[int]) -> tuple[int, ...]:
    """Return axes sharing a length with at least one other patch axis."""

    shape = tuple(int(length) for length in shape)
    if any(length <= 0 for length in shape):
        raise ValueError(f"Patch dimensions must be positive, got {shape!r}")
    axes_by_length: dict[int, list[int]] = {}
    for axis, length in enumerate(shape):
        axes_by_length.setdefault(length, []).append(axis)
    return tuple(
        axis
        for axes in axes_by_length.values()
        if len(axes) > 1
        for axis in axes
    )


def iter_mirror_axes(allowed_axes: Sequence[int]) -> list[tuple[int, ...]]:
    """Enumerate mirror variants in combination order, including identity."""

    allowed = tuple(int(axis) for axis in allowed_axes)
    return [
        axes
        for count in range(len(allowed) + 1)
        for axes in itertools.combinations(allowed, count)
    ]


def compute_importance_map_2d(
    *,
    patch_size: tuple[int, int],
    mode: str,
    sigma_scale: float = 0.125,
) -> torch.Tensor:
    """Build the exact flat constant, Gaussian, or floored Hann blend map."""

    patch_h, patch_w = (int(value) for value in patch_size)
    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size!r}")
    mode = str(mode).strip().lower()
    if mode == "constant":
        return torch.ones((patch_h, patch_w), dtype=torch.float32)
    if mode == "hann":
        weight = torch.outer(
            torch.hann_window(patch_h, periodic=False, dtype=torch.float32),
            torch.hann_window(patch_w, periodic=False, dtype=torch.float32),
        )
        weight /= torch.clamp(weight.max(), min=torch.finfo(weight.dtype).eps)
        return torch.clamp(weight, min=0.001)
    if mode != "gaussian":
        raise ValueError(f"Unsupported flat blend mode {mode!r}")
    sigma_y = max(patch_h * float(sigma_scale), 1e-6)
    sigma_x = max(patch_w * float(sigma_scale), 1e-6)
    coords_y = torch.arange(patch_h, dtype=torch.float32) - (patch_h - 1) / 2
    coords_x = torch.arange(patch_w, dtype=torch.float32) - (patch_w - 1) / 2
    grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
    weight = torch.exp(
        -0.5 * ((grid_y / sigma_y) ** 2 + (grid_x / sigma_x) ** 2)
    )
    weight /= torch.clamp(weight.max(), min=torch.finfo(weight.dtype).eps)
    return torch.clamp(weight, min=torch.finfo(weight.dtype).eps)


def resolve_patch_stride(
    *,
    patch_size: int,
    overlap: float,
    explicit_stride: int | None,
) -> int:
    """Resolve inference overlap to stride without using training overlap."""

    patch_size = int(patch_size)
    if explicit_stride is not None:
        stride = int(explicit_stride)
        if not 1 <= stride <= patch_size:
            raise ValueError(
                f"--stride must be in [1, {patch_size}], got {stride}"
            )
        return stride
    overlap = float(overlap)
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"--overlap must be in [0, 1), got {overlap}")
    return max(1, int(round(patch_size * (1.0 - overlap))))


def center_crop_layer_indices(
    layer_indices: np.ndarray,
    *,
    output_depth: int,
) -> np.ndarray:
    """Center crop source layers, choosing the upper center for even excess."""

    indices = np.asarray(layer_indices, dtype=np.int64)
    output_depth = int(output_depth)
    if output_depth <= 0:
        raise ValueError(f"output_depth must be positive, got {output_depth}")
    if indices.size <= output_depth:
        return indices
    start = indices.size // 2 - output_depth // 2
    return indices[start : start + output_depth]


def select_layer_indices(
    depth: int,
    *,
    layer_start: int | None,
    layer_end: int | None,
    output_depth: int,
    direction: str,
) -> np.ndarray:
    """Select, clamp, center crop, and optionally reverse source layers."""

    depth = int(depth)
    start = 0 if layer_start is None else int(layer_start)
    stop = depth if layer_end is None else int(layer_end)
    if start < 0:
        start += depth
    if stop < 0:
        stop += depth
    start = max(0, start)
    stop = min(depth, stop)
    if stop <= start:
        start, stop = 0, depth
    indices = center_crop_layer_indices(
        np.arange(start, stop, dtype=np.int64),
        output_depth=output_depth,
    )
    if direction == "reverse":
        indices = indices[::-1]
    return indices


def _sliding_positions_1d(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, max(1, stride)))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def iter_blocks(
    image_shape: tuple[int, int],
    patch_size: int,
    stride: int,
    mask_lowres: np.ndarray | None = None,
    occupancy_scale: tuple[int, int] = (1, 1),
) -> list[Block]:
    """Schedule flat patches, including a final boundary-aligned position."""

    height, width = (int(value) for value in image_shape)
    scale_y, scale_x = (max(1, int(value)) for value in occupancy_scale)
    blocks: list[Block] = []
    for y0 in _sliding_positions_1d(height, patch_size, stride):
        valid_h = min(patch_size, height - y0)
        for x0 in _sliding_positions_1d(width, patch_size, stride):
            valid_w = min(patch_size, width - x0)
            if mask_lowres is not None:
                low_y0 = y0 // scale_y
                low_x0 = x0 // scale_x
                low_y1 = max(low_y0 + 1, math.ceil((y0 + valid_h) / scale_y))
                low_x1 = max(low_x0 + 1, math.ceil((x0 + valid_w) / scale_x))
                if not mask_lowres[low_y0:low_y1, low_x0:low_x1].any():
                    continue
            blocks.append(Block(y0, x0, valid_h, valid_w))
    return blocks


def downsample_mask_any(mask: np.ndarray, scale_y: int, scale_x: int) -> np.ndarray:
    """Any-pool a 2D foreground mask with top-left alignment."""

    mask = np.asarray(mask, dtype=bool)
    if (scale_y, scale_x) == (1, 1):
        return mask
    pad_h = (-mask.shape[0]) % scale_y
    pad_w = (-mask.shape[1]) % scale_x
    if pad_h or pad_w:
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), constant_values=False)
    return mask.reshape(
        mask.shape[0] // scale_y,
        scale_y,
        mask.shape[1] // scale_x,
        scale_x,
    ).any(axis=(1, 3))


def choose_pyramid_array(
    root: Any,
    *,
    preferred_key: str,
    purpose: str,
) -> tuple[str, Any]:
    """Choose an array, preferring the nearest numeric pyramid level."""

    if hasattr(root, "shape"):
        return "0", root
    available = [str(key) for key in root.array_keys()]
    if preferred_key not in available:
        try:
            requested = root[preferred_key]
        except (KeyError, FileNotFoundError):
            requested = None
        if requested is not None and hasattr(requested, "shape"):
            return preferred_key, requested
    if not available:
        raise ValueError(f"No arrays found in Zarr group for {purpose}")
    if preferred_key in available:
        return preferred_key, root[preferred_key]
    numeric = [key for key in available if key.isdigit()]
    if numeric and preferred_key.isdigit():
        preferred = int(preferred_key)
        selected = min(
            numeric,
            key=lambda key: (abs(int(key) - preferred), int(key)),
        )
    else:
        selected = sorted(available)[0]
    return selected, root[selected]


def compute_nonempty_mask_from_lowres_array(array: Any) -> np.ndarray:
    """Collapse a 2D/3D low-resolution array to a 2D occupancy mask."""

    shape = tuple(int(value) for value in array.shape)
    values = np.asarray(array[:])
    if len(shape) == 2:
        return values != 0
    if len(shape) != 3:
        raise ValueError(f"Occupancy array must be rank 2 or 3, got {shape!r}")
    depth_axis = 0 if int(np.argmin(shape)) == 0 else 2
    return np.any(values != 0, axis=depth_axis)


def build_lowres_block_mask(
    root: Any,
    *,
    height: int,
    width: int,
    user_mask: np.ndarray | None,
) -> tuple[np.ndarray | None, tuple[int, int], str | None]:
    """Combine group occupancy and user mask at occupancy resolution."""

    if hasattr(root, "shape"):
        occupancy = None
        occupancy_level = None
        occupancy_h, occupancy_w = height, width
    else:
        occupancy_level, occupancy_array = choose_pyramid_array(
            root,
            preferred_key=DEFAULT_OCCUPANCY_SCAN_LEVEL,
            purpose="occupancy scan",
        )
        occupancy = compute_nonempty_mask_from_lowres_array(occupancy_array)
        occupancy_h, occupancy_w = occupancy.shape
    scale_y = max(1, int(round(height / max(1, occupancy_h))))
    scale_x = max(1, int(round(width / max(1, occupancy_w))))
    if user_mask is not None:
        pooled = downsample_mask_any(user_mask, scale_y, scale_x)
        occupancy = pooled if occupancy is None else np.logical_and(occupancy, pooled)
    return occupancy, (scale_y, scale_x), occupancy_level


class FlatPatchReader:
    """Read H/W/Z patches and apply only the legacy pre-quantization step."""

    def __init__(
        self,
        *,
        input_path: str | Path,
        resolution: str,
        depth_axis_first: bool,
        height: int,
        width: int,
        layer_indices: np.ndarray,
        output_depth: int,
        preprocessing: str,
        auth_json_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        cache_max_gb: float | None = None,
    ) -> None:
        self.input_path = input_path
        self.resolution = str(resolution)
        self.depth_axis_first = bool(depth_axis_first)
        self.height = int(height)
        self.width = int(width)
        self.layer_indices = np.asarray(layer_indices, dtype=np.int64)
        self.output_depth = int(output_depth)
        self.preprocessing = str(preprocessing)
        self.auth_json_path = auth_json_path
        self.cache_dir = cache_dir
        self.cache_max_gb = cache_max_gb
        if self.layer_indices.size == 0:
            raise ValueError("Flat inference selected no source layers")
        if self.output_depth < self.layer_indices.size:
            raise ValueError(
                f"output_depth={self.output_depth} is smaller than selected "
                f"depth={self.layer_indices.size}"
            )
        self.output_depth_start = (
            self.output_depth - self.layer_indices.size
        ) // 2
        self._array = None
        self._read_mode = "fancy"
        self._z_start = int(self.layer_indices[0])
        self._z_stop = int(self.layer_indices[-1]) + 1
        if self.layer_indices.size > 1 and np.all(np.diff(self.layer_indices) == 1):
            self._read_mode = "ascending"
        elif self.layer_indices.size > 1 and np.all(np.diff(self.layer_indices) == -1):
            self._read_mode = "descending"
            self._z_start = int(self.layer_indices[-1])
            self._z_stop = int(self.layer_indices[0]) + 1

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None
        return state

    def _ensure_array(self):
        if self._array is None:
            self._array = open_volume(
                self.input_path,
                self.resolution,
                self.auth_json_path,
                cache_dir=self.cache_dir,
                cache_max_gb=self.cache_max_gb,
            )
        return self._array

    def _read_raw(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        array = self._ensure_array()
        if self.depth_axis_first:
            if self._read_mode == "ascending":
                block = array[self._z_start : self._z_stop, y0:y1, x0:x1]
            elif self._read_mode == "descending":
                block = array[self._z_start : self._z_stop, y0:y1, x0:x1][::-1]
            else:
                block = array[self.layer_indices, y0:y1, x0:x1]
            return np.transpose(np.asarray(block), (1, 2, 0))
        if self._read_mode == "ascending":
            block = array[y0:y1, x0:x1, self._z_start : self._z_stop]
        elif self._read_mode == "descending":
            block = array[y0:y1, x0:x1, self._z_start : self._z_stop][..., ::-1]
        else:
            block = array[y0:y1, x0:x1, self.layer_indices]
        return np.asarray(block)

    def read(self, y0: int, x0: int, out_h: int, out_w: int) -> np.ndarray:
        """Read one padded H/W/Z patch with centered short-depth placement."""

        y1, x1 = y0 + out_h, x0 + out_w
        source_y0, source_x0 = max(0, y0), max(0, x0)
        source_y1, source_x1 = min(self.height, y1), min(self.width, x1)
        dtype = np.float32 if self.preprocessing == "tifxyz_robust" else np.uint8
        output = np.zeros((out_h, out_w, self.output_depth), dtype=dtype)
        if source_y1 <= source_y0 or source_x1 <= source_x0:
            return output
        block = self._read_raw(source_y0, source_y1, source_x0, source_x1)
        if self.preprocessing == "tifxyz_robust":
            block = np.asarray(block, dtype=np.float32)
        elif self.preprocessing == "legacy_uint8":
            block = convert_volume_dtype(block)
            np.clip(block, 0, 200, out=block)
        elif self.preprocessing == "divide_255":
            block = np.asarray(block)
            if block.dtype != np.uint8:
                raise TypeError(
                    "Flat divide preprocessing requires uint8 input, got "
                    f"{block.dtype}"
                )
            block = np.ascontiguousarray(block)
        else:
            raise ValueError(f"Unsupported flat preprocessing {self.preprocessing!r}")
        depth_start = self.output_depth_start
        output[
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
            depth_start : depth_start + self.layer_indices.size,
        ] = block
        return output


class FlatBlockDataset(Dataset):
    """Turn scheduled flat blocks into C/Z/Y/X model tensors and metadata."""

    def __init__(
        self,
        *,
        reader: FlatPatchReader,
        blocks: Sequence[Block],
        patch_size: int,
        preprocessing: str,
    ) -> None:
        self.reader = reader
        self.blocks = tuple(blocks)
        self.patch_size = int(patch_size)
        self.preprocessing = str(preprocessing)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int):
        block = self.blocks[index]
        patch_HWZ = self.reader.read(
            block.y0,
            block.x0,
            self.patch_size,
            self.patch_size,
        )
        nonempty = int(patch_HWZ.any())
        patch_ZYX = np.moveaxis(patch_HWZ, -1, 0)
        patch_ZYX = normalize_flat_patch(patch_ZYX, self.preprocessing)
        image_CZYX = torch.from_numpy(np.ascontiguousarray(patch_ZYX)).unsqueeze(0)
        metadata = torch.tensor(
            [block.y0, block.x0, block.valid_h, block.valid_w, nonempty],
            dtype=torch.int64,
        )
        return image_CZYX, metadata


def flip_tensor_for_patch_axes(
    tensor: torch.Tensor, patch_axes: Sequence[int]
) -> torch.Tensor:
    """Flip BCZYX input over selected Z/Y/X patch axes."""

    axes = tuple(int(axis) for axis in patch_axes)
    return tensor if not axes else torch.flip(tensor, [axis + 2 for axis in axes])


def unflip_spatial_output_for_patch_axes(
    output: torch.Tensor, patch_axes: Sequence[int]
) -> torch.Tensor:
    """Restore only Y/X axes of a BCYX flat output."""

    dimensions = [
        2 if int(axis) == 1 else 3
        for axis in patch_axes
        if int(axis) in {1, 2}
    ]
    return output if not dimensions else torch.flip(output, dimensions)


def logits_to_probabilities(
    logits: torch.Tensor,
    *,
    image_hw: tuple[int, int],
) -> torch.Tensor:
    """Apply sigmoid and bilinearly restore the flat patch size."""

    if logits.ndim != 4:
        raise ValueError(
            f"Flat model logits must have shape [B,C,H,W], got {tuple(logits.shape)}"
        )
    probabilities = logits.float().sigmoid_()
    if probabilities.shape[-2:] != image_hw:
        probabilities = F.interpolate(
            probabilities,
            size=image_hw,
            mode="bilinear",
            align_corners=False,
        )
    return probabilities


def predict_with_mirror_tta(
    model: nn.Module,
    images_BCZYX: torch.Tensor,
    *,
    tta_axes: Sequence[int],
    tta_batch_size: int | None,
) -> torch.Tensor:
    """Average flat mirror variants in stable combination order."""

    variants = iter_mirror_axes(tta_axes)
    if len(variants) == 1:
        return logits_to_probabilities(
            model(images_BCZYX),
            image_hw=tuple(int(value) for value in images_BCZYX.shape[-2:]),
        )
    original_batch = int(images_BCZYX.shape[0])
    per_forward = (
        len(variants)
        if tta_batch_size is None
        else min(int(tta_batch_size), len(variants))
    )
    probability_sum: torch.Tensor | None = None
    for start in range(0, len(variants), per_forward):
        selected = variants[start : start + per_forward]
        variant_images = torch.cat(
            [flip_tensor_for_patch_axes(images_BCZYX, axes) for axes in selected]
        )
        probabilities_flat = logits_to_probabilities(
            model(variant_images),
            image_hw=tuple(int(value) for value in images_BCZYX.shape[-2:]),
        )
        probabilities_by_variant = probabilities_flat.reshape(
            len(selected), original_batch, *probabilities_flat.shape[1:]
        )
        for variant_index, axes in enumerate(selected):
            restored = unflip_spatial_output_for_patch_axes(
                probabilities_by_variant[variant_index], axes
            )
            probability_sum = (
                restored if probability_sum is None else probability_sum + restored
            )
    if probability_sum is None:
        raise RuntimeError("Mirror TTA produced no variants")
    return probability_sum / float(len(variants))


def iter_overlapping_chunks(
    y0: int,
    x0: int,
    valid_h: int,
    valid_w: int,
    chunk_shape: tuple[int, int],
) -> Iterator[ChunkKey]:
    """Yield output chunks touched by one valid flat tile."""

    chunk_h, chunk_w = (max(1, int(value)) for value in chunk_shape)
    final_y = y0 + max(1, valid_h) - 1
    final_x = x0 + max(1, valid_w) - 1
    for row in range(y0 // chunk_h, final_y // chunk_h + 1):
        for col in range(x0 // chunk_w, final_x // chunk_w + 1):
            yield ChunkKey(row, col)


def compute_chunk_contribution_counts(
    blocks: Sequence[Block],
    *,
    chunk_shape: tuple[int, int],
) -> dict[ChunkKey, int]:
    """Count scheduled contributions so completed chunks can flush early."""

    counts: dict[ChunkKey, int] = {}
    for block in blocks:
        for key in iter_overlapping_chunks(
            block.y0, block.x0, block.valid_h, block.valid_w, chunk_shape
        ):
            counts[key] = counts.get(key, 0) + 1
    return counts


class ChunkAccumulator:
    """Accumulate float32 probability and weight sums by output chunk."""

    def __init__(
        self,
        *,
        shape: tuple[int, int],
        chunk_shape: tuple[int, int],
        prob_sum_store: Any,
        weight_sum_store: Any,
        contribution_counts: Mapping[ChunkKey, int],
    ) -> None:
        self.height, self.width = (int(value) for value in shape)
        self.chunk_h, self.chunk_w = (
            max(1, int(value)) for value in chunk_shape
        )
        self.prob_sum_store = prob_sum_store
        self.weight_sum_store = weight_sum_store
        self.contribution_counts = dict(contribution_counts)
        self.seen_counts: dict[ChunkKey, int] = {}
        self.buffers: dict[ChunkKey, tuple[np.ndarray, np.ndarray]] = {}

    def _bounds(self, key: ChunkKey) -> tuple[int, int, int, int]:
        y0, x0 = key.row * self.chunk_h, key.col * self.chunk_w
        return y0, min(self.height, y0 + self.chunk_h), x0, min(
            self.width, x0 + self.chunk_w
        )

    def _buffers(self, key: ChunkKey) -> tuple[np.ndarray, np.ndarray]:
        if key not in self.buffers:
            y0, y1, x0, x1 = self._bounds(key)
            self.buffers[key] = (
                np.zeros((y1 - y0, x1 - x0), dtype=np.float32),
                np.zeros((y1 - y0, x1 - x0), dtype=np.float32),
            )
        return self.buffers[key]

    def add_tile(
        self,
        *,
        y0: int,
        x0: int,
        tile: np.ndarray,
        tile_weights: np.ndarray,
    ) -> None:
        """Add one weighted tile and flush every newly completed chunk."""

        valid_h, valid_w = tile.shape
        y1, x1 = y0 + valid_h, x0 + valid_w
        for key in iter_overlapping_chunks(
            y0, x0, valid_h, valid_w, (self.chunk_h, self.chunk_w)
        ):
            chunk_y0, chunk_y1, chunk_x0, chunk_x1 = self._bounds(key)
            iy0, iy1 = max(y0, chunk_y0), min(y1, chunk_y1)
            ix0, ix1 = max(x0, chunk_x0), min(x1, chunk_x1)
            if iy1 <= iy0 or ix1 <= ix0:
                continue
            probability_buffer, weight_buffer = self._buffers(key)
            source = (
                slice(iy0 - y0, iy1 - y0),
                slice(ix0 - x0, ix1 - x0),
            )
            destination = (
                slice(iy0 - chunk_y0, iy1 - chunk_y0),
                slice(ix0 - chunk_x0, ix1 - chunk_x0),
            )
            weights = tile_weights[source]
            probability_buffer[destination] += tile[source] * weights
            weight_buffer[destination] += weights
            seen = self.seen_counts.get(key, 0) + 1
            if seen >= self.contribution_counts[key]:
                self._flush(key)
            else:
                self.seen_counts[key] = seen

    def _flush(self, key: ChunkKey) -> None:
        probability, weight = self.buffers.pop(key)
        self.seen_counts.pop(key, None)
        y0, y1, x0, x1 = self._bounds(key)
        self.prob_sum_store[y0:y1, x0:x1] = probability
        self.weight_sum_store[y0:y1, x0:x1] = weight

    def flush_remaining(self) -> None:
        """Flush chunks whose skipped raw-empty blocks prevented early completion."""

        for key in tuple(self.buffers):
            self._flush(key)


def run_block_inference(
    *,
    loader: DataLoader,
    model: nn.Module,
    accumulator: ChunkAccumulator,
    weight_map: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    tta_axes: Sequence[int],
    tta_batch_size: int | None,
) -> None:
    """Forward nonempty flat patches and feed their weighted probabilities."""

    mask_f32 = None if mask is None else mask.astype(np.float32, copy=False)
    autocast = (
        torch.autocast("cuda", dtype=amp_dtype)
        if device.type == "cuda"
        else nullcontext()
    )
    with torch.inference_mode(), autocast:
        for images_BCZYX, metadata in loader:
            keep = metadata[:, 4] > 0
            if not bool(keep.any()):
                continue
            images_BCZYX = images_BCZYX[keep].to(device, non_blocking=True)
            metadata = metadata[keep]
            if tta_axes:
                probabilities = predict_with_mirror_tta(
                    model,
                    images_BCZYX,
                    tta_axes=tta_axes,
                    tta_batch_size=tta_batch_size,
                )
            else:
                probabilities = logits_to_probabilities(
                    model(images_BCZYX),
                    image_hw=tuple(int(value) for value in images_BCZYX.shape[-2:]),
                )
            probabilities_np = probabilities.cpu().numpy()[:, 0]
            for probability, values in zip(probabilities_np, metadata.numpy()):
                y0, x0, valid_h, valid_w = (int(value) for value in values[:4])
                tile = probability[:valid_h, :valid_w]
                weights = weight_map[:valid_h, :valid_w]
                if mask_f32 is not None:
                    weights = weights * mask_f32[
                        y0 : y0 + valid_h, x0 : x0 + valid_w
                    ]
                accumulator.add_tile(
                    y0=y0,
                    x0=x0,
                    tile=tile,
                    tile_weights=weights,
                )


def iter_probability_tiles(
    prob_sum_store: Any,
    weight_sum_store: Any,
    tile_shape: tuple[int, int],
) -> Iterator[np.ndarray]:
    """Encode normalized chunks by clipping, scaling, and truncating to uint8."""

    height, width = (int(value) for value in prob_sum_store.shape)
    tile_h, tile_w = (int(value) for value in tile_shape)
    for y0 in range(0, height, tile_h):
        for x0 in range(0, width, tile_w):
            y1, x1 = min(height, y0 + tile_h), min(width, x0 + tile_w)
            probability = np.asarray(
                prob_sum_store[y0:y1, x0:x1], dtype=np.float32
            )
            weight = np.asarray(
                weight_sum_store[y0:y1, x0:x1], dtype=np.float32
            )
            np.divide(probability, weight, out=probability, where=weight > 1e-6)
            probability[weight <= 1e-6] = 0
            np.clip(probability, 0, 1, out=probability)
            probability *= 255
            yield probability.astype(np.uint8, copy=False)


def write_output_tiff(
    prob_sum_store: Any,
    weight_sum_store: Any,
    output_path: Path,
    tile_shape: tuple[int, int],
) -> None:
    """Write one tiled, LZW, BigTIFF-compatible flat probability image."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        output_path,
        iter_probability_tiles(prob_sum_store, weight_sum_store, tile_shape),
        shape=tuple(int(value) for value in prob_sum_store.shape),
        dtype=np.uint8,
        compression="lzw",
        tile=tuple(int(value) for value in tile_shape),
        bigtiff=True,
        metadata=None,
        software="vesuvius.ink_detection.inference.infer",
    )


def open_temp_zarr_array(
    path: Path,
    *,
    shape: tuple[int, int],
    chunks: tuple[int, int],
):
    """Create one explicit-v2 float32 accumulation array."""

    try:
        return zarr.open(
            str(path),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=np.float32,
            zarr_version=2,
        )
    except TypeError:
        return zarr.open(
            str(path),
            mode="w",
            shape=shape,
            chunks=chunks,
            dtype=np.float32,
        )


def load_grayscale_mask(path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Read a nonzero foreground TIFF with top-left crop/pad alignment."""

    image = np.squeeze(np.asarray(tifffile.imread(path)))
    if image.ndim == 3:
        image = image[..., 0]
    if image.ndim != 2:
        raise ValueError(f"Mask must be 2D, got {image.shape!r} from {path}")
    output = np.zeros(target_shape, dtype=bool)
    height = min(target_shape[0], image.shape[0])
    width = min(target_shape[1], image.shape[1])
    output[:height, :width] = image[:height, :width] != 0
    return output


def load_flat_inference_state(
    model: nn.Module, state: Mapping[str, Any]
):
    """Load a flat inference state permissively after uniform DDP unwrapping."""

    if state and all(str(key).startswith("module.") for key in state):
        state = {
            str(key)[len("module.") :]: value for key, value in state.items()
        }
    return model.load_state_dict(state, strict=False)


def configure_model(args: Any) -> ConfiguredModel:
    """Rebuild one strict flat model and preprocessing contract from checkpoint."""

    if args.model_type == "resnet3d":
        raise ValueError("Unsupported model_type 'resnet3d' for ink inference")
    payload = load_checkpoint(args.checkpoint)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("config"), Mapping
    ):
        raise ValueError(
            f"Inference checkpoint {str(args.checkpoint)!r} requires a config mapping"
        )
    resolved_mapping = resolve_pretrained_backbone_config(
        payload["config"], checkpoint_path=args.checkpoint
    )
    config = InkConfig.from_mapping(resolved_mapping)
    if config.data.mode != "flat":
        raise ValueError(
            f"Flat inference requires checkpoint mode='flat', got {config.data.mode!r}"
        )
    selected_state, state = select_inference_weights(
        payload, prefer_ema=True, source=args.checkpoint
    )
    base_model = make_model(config)
    incompatibility = load_flat_inference_state(base_model, state)
    LOGGER.info(
        "Loaded %s weights from %s (missing_keys=%d unexpected_keys=%d)",
        selected_state,
        args.checkpoint,
        len(incompatibility.missing_keys),
        len(incompatibility.unexpected_keys),
    )
    if args.metadata_json is not None:
        LOGGER.info(
            "Ignoring --metadata-json for config-backed checkpoint %s",
            args.checkpoint,
        )
    base_model.eval()
    crop_z, crop_y, crop_x = config.model.crop_size
    if crop_y != crop_x:
        raise ValueError(
            f"Flat inference requires square Y/X patches, got {config.model.crop_size!r}"
        )
    model = FlatTargetModel(
        base_model,
        input_pad_depth_to=config.model.input_pad_depth_to,
    ).eval()
    return ConfiguredModel(
        model=model,
        patch_size=crop_y,
        input_depth=crop_z,
        preprocessing=flat_preprocessing_from_config(config.data.normalization),
        amp_dtype=resolve_amp_dtype(args.amp_dtype, payload, args.checkpoint),
        config=config,
        selected_state=selected_state,
    )


def _volume_axes(array: Any) -> tuple[bool, int, int, int, int, int]:
    shape = tuple(int(value) for value in array.shape)
    if len(shape) != 3:
        raise ValueError(f"Flat input array must be rank 3, got {shape!r}")
    chunks = tuple(int(value) for value in (array.chunks or shape))
    depth_first = int(np.argmin(shape)) == 0
    if depth_first:
        depth, height, width = shape
        _, chunk_h, chunk_w = chunks
    else:
        height, width, depth = shape
        chunk_h, chunk_w, _ = chunks
    return depth_first, depth, height, width, chunk_h, chunk_w


def infer_single_zarr(
    *,
    args: Any,
    input_zarr: str | Path,
    configured_model: ConfiguredModel,
    device: torch.device,
    output_tiff: Path,
    layer_direction: str = "forward",
) -> None:
    """Run one direction over one surface-volume input and replace a TIFF."""

    config = configured_model.config
    root = open_volume_root(
        input_zarr,
        config.data.volume_auth_json,
        cache_dir=config.data.volume_cache_dir,
        cache_max_gb=config.data.volume_cache_max_gb,
    )
    resolution = "0" if hasattr(root, "shape") else str(args.resolution)
    volume = select_volume_level(root, resolution, source=str(input_zarr))
    depth_first, depth, height, width, chunk_h, chunk_w = _volume_axes(volume)
    patch_size = configured_model.patch_size
    stride = resolve_patch_stride(
        patch_size=patch_size,
        overlap=args.overlap,
        explicit_stride=args.stride,
    )
    blend_mode = str(args.blend_mode)
    if blend_mode == "auto":
        blend_mode = "constant" if stride >= patch_size else "gaussian"
    tile_shape = (patch_size, patch_size)
    if patch_size % 16:
        tile_shape = (chunk_h, chunk_w)
    layer_indices = select_layer_indices(
        depth,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
        output_depth=configured_model.input_depth,
        direction=layer_direction,
    )
    reader = FlatPatchReader(
        input_path=input_zarr,
        resolution=resolution,
        depth_axis_first=depth_first,
        height=height,
        width=width,
        layer_indices=layer_indices,
        output_depth=configured_model.input_depth,
        preprocessing=configured_model.preprocessing,
        auth_json_path=config.data.volume_auth_json,
        cache_dir=config.data.volume_cache_dir,
        cache_max_gb=config.data.volume_cache_max_gb,
    )
    mask = (
        None
        if args.mask_path is None
        else load_grayscale_mask(args.mask_path, (height, width))
    )
    occupancy, occupancy_scale, _ = build_lowres_block_mask(
        root,
        height=height,
        width=width,
        user_mask=mask,
    )
    blocks = iter_blocks(
        (height, width),
        patch_size,
        stride,
        occupancy,
        occupancy_scale,
    )
    dataset = FlatBlockDataset(
        reader=reader,
        blocks=blocks,
        patch_size=patch_size,
        preprocessing=configured_model.preprocessing,
    )
    effective_batch_size = args.batch_size * max(1, len(args.gpu_ids))
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": effective_batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
            multiprocessing_context="spawn",
        )
    loader = DataLoader(**loader_kwargs)
    weight_map = compute_importance_map_2d(
        patch_size=(patch_size, patch_size), mode=blend_mode
    ).numpy()
    tta_axes = (
        compute_equal_length_mirror_axes(
            (configured_model.input_depth, patch_size, patch_size)
        )
        if args.tta_mirror
        else ()
    )

    temporary = Path(tempfile.mkdtemp(prefix="ink_flat_infer_"))
    try:
        accumulation_chunks = (
            min(tile_shape[0], height),
            min(tile_shape[1], width),
        )
        probability_sum = open_temp_zarr_array(
            temporary / "probability.zarr",
            shape=(height, width),
            chunks=accumulation_chunks,
        )
        weight_sum = open_temp_zarr_array(
            temporary / "weight.zarr",
            shape=(height, width),
            chunks=accumulation_chunks,
        )
        accumulator = ChunkAccumulator(
            shape=(height, width),
            chunk_shape=accumulation_chunks,
            prob_sum_store=probability_sum,
            weight_sum_store=weight_sum,
            contribution_counts=compute_chunk_contribution_counts(
                blocks, chunk_shape=accumulation_chunks
            ),
        )
        if blocks:
            run_block_inference(
                loader=loader,
                model=configured_model.model,
                accumulator=accumulator,
                weight_map=weight_map,
                mask=mask,
                device=device,
                amp_dtype=configured_model.amp_dtype,
                tta_axes=tta_axes,
                tta_batch_size=args.tta_batch_size,
            )
            accumulator.flush_remaining()
        write_output_tiff(
            probability_sum,
            weight_sum,
            output_tiff,
            tile_shape,
        )
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def resolve_run_directions(direction: str) -> tuple[str, ...]:
    """Expand the CLI direction to concrete ordered runs."""

    return ("forward", "reverse") if direction == "both" else (direction,)


def resolve_single_output_path(
    output_tiff: Path,
    *,
    direction: str,
    requested_direction: str,
) -> Path:
    """Append only the reverse suffix for a two-direction single run."""

    if requested_direction != "both" or direction == "forward":
        return output_tiff
    return output_tiff.with_name(
        f"{output_tiff.stem}_{direction}{output_tiff.suffix}"
    )


def resolve_segment_zarr_path(segment_dir: Path) -> Path:
    """Resolve the frozen folder-mode direct candidates or sole discovery."""

    candidates = (
        segment_dir / segment_dir.name,
        segment_dir / f"{segment_dir.name}.zarr",
        segment_dir / f"{segment_dir.name}.ome.zarr",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    discovered = []
    for child in sorted(segment_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.suffix == ".zarr" or any(
            (child / marker).exists()
            for marker in (".zgroup", ".zarray", "zarr.json")
        ):
            discovered.append(child)
    if len(discovered) == 1:
        return discovered[0]
    raise FileNotFoundError(
        f"Could not resolve exactly one Zarr for segment {segment_dir.name!r}"
    )


def infer_folder(
    args: Any,
    configured_model: ConfiguredModel,
    *,
    device: torch.device,
) -> None:
    """Run sorted folder segments, skipping any previously dated prediction."""

    folder = Path(args.folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"--folder is not a directory: {folder}")
    try:
        resolve_segment_zarr_path(folder)
    except FileNotFoundError:
        segment_dirs = sorted(path for path in folder.iterdir() if path.is_dir())
        if not segment_dirs:
            raise FileNotFoundError(f"No segment directories found under {folder}")
    else:
        segment_dirs = [folder]
    checkpoint_stem = Path(args.checkpoint).stem
    date = datetime.now().strftime("%d%m%y")
    prefix = f"{args.output_prefix}_" if args.output_prefix else ""
    ran_count = 0
    skipped_count = 0
    for segment_dir in segment_dirs:
        try:
            input_zarr = resolve_segment_zarr_path(segment_dir)
        except FileNotFoundError as exc:
            LOGGER.warning("Skipping %s: %s", segment_dir, exc)
            skipped_count += 1
            continue
        for direction in resolve_run_directions(args.direction):
            name_prefix = (
                f"{prefix}{segment_dir.name}_{checkpoint_stem}_{direction}_"
            )
            prediction_dir = segment_dir / "preds"
            existing = (
                next(prediction_dir.glob(f"{name_prefix}*.tif"), None)
                if prediction_dir.exists()
                else None
            )
            if existing is not None:
                LOGGER.info(
                    "Skipping segment=%s direction=%s — already have %s",
                    segment_dir.name,
                    direction,
                    existing.name,
                )
                skipped_count += 1
                continue
            infer_single_zarr(
                args=args,
                input_zarr=input_zarr,
                configured_model=configured_model,
                device=device,
                output_tiff=prediction_dir / f"{name_prefix}{date}.tif",
                layer_direction=direction,
            )
            ran_count += 1
    LOGGER.info(
        "Folder run complete. segments_ran=%d segments_skipped=%d",
        ran_count,
        skipped_count,
    )


def _is_url(path: str | Path) -> bool:
    parsed = urlparse(str(path))
    return bool(parsed.scheme and parsed.netloc)


def normalize_inference_paths(args: Any) -> Any:
    """Normalize the two accepted positional/folder command shapes."""

    if args.input_zarr is not None and not _is_url(args.input_zarr):
        args.input_zarr = Path(args.input_zarr)
    args.output_tiff = None if args.output_tiff is None else Path(args.output_tiff)
    args.folder = None if args.folder is None else Path(args.folder)
    args.checkpoint = None if args.checkpoint is None else Path(args.checkpoint)
    if args.checkpoint_path is not None:
        args.checkpoint = Path(args.checkpoint_path)
    if (
        args.folder is not None
        and args.checkpoint is None
        and args.input_zarr is not None
        and args.output_tiff is None
    ):
        args.checkpoint = Path(args.input_zarr)
        args.input_zarr = None
    if args.checkpoint is None:
        raise ValueError(
            "Checkpoint is required as a positional or --checkpoint-path"
        )
    if args.folder is not None:
        if args.output_tiff is not None:
            raise ValueError("output_tiff is forbidden with --folder")
    elif args.input_zarr is None or args.output_tiff is None:
        raise ValueError(
            "Single mode requires <input_zarr> <checkpoint> <output_tiff>"
        )
    return args


def parse_args(argv: Sequence[str] | None = None):
    """Parse the frozen flat-inference CLI with separator aliases."""

    parser = HyphenUnderscoreParser(
        description="Run ink inference on a flat surface-volume Zarr"
    )
    parser.add_argument("input_zarr", nargs="?")
    parser.add_argument("checkpoint", nargs="?", type=Path)
    parser.add_argument("output_tiff", nargs="?", type=Path)
    parser.add_argument(
        "--model-type",
        choices=("auto", "resnet3d", "residual_unet", "tifxyz_unet"),
        default="auto",
    )
    parser.add_argument("--folder", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--metadata-json", type=Path)
    parser.add_argument("--mask-path", type=Path)
    parser.add_argument("--resolution", default="0")
    parser.add_argument(
        "--num-workers", "--workers", dest="num_workers", type=int, default=4
    )
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--stride", type=int)
    parser.add_argument(
        "--blend-mode",
        choices=("auto", "constant", "gaussian", "hann"),
        default="auto",
    )
    parser.add_argument("--layer-start", type=int)
    parser.add_argument("--layer-end", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--direction", choices=("forward", "reverse", "both"), default="forward"
    )
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "default", "fp16", "bf16"),
        default="auto",
    )
    parser.add_argument("--tta-mirror", action="store_true")
    parser.add_argument("--tta-batch-size", type=int)
    parser.add_argument("--gpus")
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=True)
    args = parser.parse_args(argv)
    if args.num_workers < 0:
        parser.error("--num-workers must be nonnegative")
    if args.prefetch_factor <= 0:
        parser.error("--prefetch-factor must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.tta_batch_size is not None and args.tta_batch_size <= 0:
        parser.error("--tta-batch-size must be positive")
    if not 0 <= args.overlap < 1:
        parser.error("--overlap must be in [0, 1)")
    if args.stride is not None and args.stride <= 0:
        parser.error("--stride must be positive")
    try:
        args.gpu_ids = parse_gpu_ids(args.gpus)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Run the flat inference command."""

    args = normalize_inference_paths(parse_args(argv))
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    configured = configure_model(args)
    prepared_model, device, _ = prepare_model_for_inference(
        configured.model,
        gpu_ids=args.gpu_ids,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
    )
    configured = ConfiguredModel(
        model=prepared_model,
        patch_size=configured.patch_size,
        input_depth=configured.input_depth,
        preprocessing=configured.preprocessing,
        amp_dtype=configured.amp_dtype,
        config=configured.config,
        selected_state=configured.selected_state,
    )
    if args.folder is not None:
        infer_folder(args, configured, device=device)
    else:
        for direction in resolve_run_directions(args.direction):
            infer_single_zarr(
                args=args,
                input_zarr=args.input_zarr,
                configured_model=configured,
                device=device,
                output_tiff=resolve_single_output_path(
                    args.output_tiff,
                    direction=direction,
                    requested_direction=args.direction,
                ),
                layer_direction=direction,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
