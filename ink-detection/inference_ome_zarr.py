#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import math
import shutil
import tempfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
import zarr
from monai.data.utils import compute_importance_map
from monai.inferers import SlidingWindowInferer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


LOGGER = logging.getLogger("inference_ome_zarr")
DEFAULT_OCCUPANCY_LEVEL = "4"
DEFAULT_OVERLAP = 0.25
DEFAULT_SW_BATCH_SIZE = 4
DEFAULT_PREFETCH_FACTOR = 2


@dataclass(frozen=True)
class Block:
    y0: int
    x0: int
    valid_h: int
    valid_w: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple MONAI sliding-window inference for OME-Zarr volumes."
    )
    parser.add_argument("input_zarr", nargs="?", help="Input OME-Zarr path or URL.")
    parser.add_argument("checkpoint", type=Path, nargs="?", help="Model checkpoint path.")
    parser.add_argument("output_tiff", type=Path, nargs="?", help="Output uint8 tiled TIFF path.")
    parser.add_argument("--model-type", choices=("resnet3d", "residual_unet"), required=True)
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
    parser.add_argument("--metadata-json", type=Path, default=None)
    parser.add_argument("--mask-path", type=Path, default=None)
    parser.add_argument("--resolution", default="0")
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP)
    parser.add_argument("--layer-start", type=int, default=None)
    parser.add_argument("--layer-end", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--reverse-layers", "--reverse", dest="reverse_layers", action="store_true")
    return parser.parse_args(argv)


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

    for y0 in _sliding_positions_1d(image_h, patch_h, stride):
        valid_h = min(patch_h, image_h - y0)
        for x0 in _sliding_positions_1d(image_w, patch_w, stride):
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
    ):
        self.input_path = input_path if is_url_like_path(input_path) else Path(input_path)
        self.resolution = str(resolution)
        self.depth_axis_first = bool(depth_axis_first)
        self.height = int(height)
        self.width = int(width)
        self.layer_indices = np.asarray(layer_indices, dtype=np.int64)
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

        out = np.zeros((out_h, out_w, self.layer_indices.size), dtype=np.uint8)
        if yy1 <= yy0 or xx1 <= xx0:
            return out

        block = self._read_raw(yy0, yy1, xx0, xx1)
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


class OmeZarrBlockDataset(Dataset):
    def __init__(
        self,
        *,
        reader: OmeZarrPatchReader,
        blocks: Sequence[Block],
        patch_size: int,
    ):
        self.reader = reader
        self.blocks = list(blocks)
        self.patch_h = int(patch_size)
        self.patch_w = int(patch_size)

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, index: int):
        block = self.blocks[index]
        patch = self.reader.read(block.y0, block.x0, self.patch_h, self.patch_w)
        patch = patch.astype(np.float32, copy=False) / 255.0
        patch = np.moveaxis(patch, -1, 0)
        image = torch.from_numpy(np.ascontiguousarray(patch))
        meta = torch.tensor([block.y0, block.x0, block.valid_h, block.valid_w], dtype=torch.int64)
        return image, meta


def configure_model(args: argparse.Namespace):
    from train_resnet3d_lib.checkpointing import load_state_dict_from_checkpoint
    from train_resnet3d_lib.config import CFG, apply_metadata_hyperparameters, load_metadata_json
    from train_resnet3d_lib.model import RegressionPLModel

    metadata = None
    if args.metadata_json is not None:
        metadata = load_metadata_json(str(args.metadata_json))
        apply_metadata_hyperparameters(CFG, metadata)
    elif args.model_type == "residual_unet":
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

    state_dict = load_state_dict_from_checkpoint(str(args.checkpoint))
    incompat = model.load_state_dict(state_dict, strict=False)
    LOGGER.info(
        "Loaded checkpoint %s (missing_keys=%d unexpected_keys=%d)",
        args.checkpoint,
        len(incompat.missing_keys),
        len(incompat.unexpected_keys),
    )
    model.eval()
    return model, CFG, metadata


def run_block_inference(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    inferer: SlidingWindowInferer,
    prob_sum_store,
    weight_sum_store,
    weight_map: np.ndarray,
    mask: np.ndarray | None,
    device: torch.device,
    amp: bool,
) -> None:
    autocast_enabled = bool(amp and device.type == "cuda")
    logged_resize = False
    with torch.inference_mode():
        for images, meta in tqdm(loader, desc="Infer", unit="block"):
            images = images.to(device, non_blocking=True)
            amp_context = (
                torch.autocast(device_type="cuda", enabled=True)
                if autocast_enabled
                else nullcontext()
            )
            with amp_context:
                logits = inferer(images, model)
            probs_t = torch.sigmoid(logits).to(dtype=torch.float32)
            if probs_t.shape[-2:] != images.shape[-2:]:
                if not logged_resize:
                    LOGGER.info(
                        "Resizing model output from %s to input patch size %s for stitching.",
                        tuple(int(v) for v in probs_t.shape[-2:]),
                        tuple(int(v) for v in images.shape[-2:]),
                    )
                    logged_resize = True
                probs_t = F.interpolate(
                    probs_t,
                    size=images.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
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
                y_slice = slice(y0, y0 + valid_h)
                x_slice = slice(x0, x0 + valid_w)
                prob_sum_store[y_slice, x_slice] = prob_sum_store[y_slice, x_slice] + (tile * tile_weights)
                weight_sum_store[y_slice, x_slice] = weight_sum_store[y_slice, x_slice] + tile_weights


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
    model: torch.nn.Module,
    cfg,
    input_zarr: str | Path,
    output_tiff: Path,
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

    if isinstance(root, zarr.Array):
        occupancy_arr = root
    else:
        occupancy_level = DEFAULT_OCCUPANCY_LEVEL
        available_levels = [str(key) for key in root.array_keys()]
        if occupancy_level not in available_levels and available_levels:
            available_levels = sorted(
                available_levels,
                key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
            )
            occupancy_level = available_levels[-1]
            LOGGER.warning(
                "Requested occupancy level %s was not found. Using level %s.",
                DEFAULT_OCCUPANCY_LEVEL,
                occupancy_level,
            )
        occupancy_arr = root[occupancy_level]

    occupancy_shape = tuple(int(v) for v in occupancy_arr.shape)
    occupancy_depth_first = int(np.argmin(occupancy_shape)) == 0
    if occupancy_depth_first:
        occupancy_h, occupancy_w = occupancy_shape[1], occupancy_shape[2]
    else:
        occupancy_h, occupancy_w = occupancy_shape[0], occupancy_shape[1]
    occupancy_scale_y = max(1, int(round(height / max(1, occupancy_h))))
    occupancy_scale_x = max(1, int(round(width / max(1, occupancy_w))))

    roi_size = int(getattr(cfg, "size", 256))
    patch_size = roi_size
    patch_stride = max(1, int(round(float(patch_size) * (1.0 - float(args.overlap)))))

    tiff_tile_shape = (patch_size, patch_size)
    if tiff_tile_shape[0] % 16 != 0 or tiff_tile_shape[1] % 16 != 0:
        tiff_tile_shape = (int(spatial_chunk_h), int(spatial_chunk_w))

    in_chans = int(getattr(cfg, "in_chans", 62))
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
    if bool(args.reverse_layers):
        layer_indices = layer_indices[::-1]
    reader = OmeZarrPatchReader(
        input_path=input_zarr,
        resolution=resolution,
        depth_axis_first=depth_axis_first,
        height=height,
        width=width,
        layer_indices=layer_indices,
    )

    layer_order = "reverse" if bool(args.reverse_layers) else "forward"
    LOGGER.info(
        "Input level=%s shape=(depth=%d, height=%d, width=%d) chunks=(%d, %d) patch=%d stride=%d blend=%.3f tiff_tile=%s in_chans=%d layer_order=%s",
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
    )

    mask = None
    mask_lowres = None
    if args.mask_path is not None:
        mask = load_grayscale_mask(args.mask_path, (height, width))
        mask_lowres = downsample_mask_any(mask, occupancy_scale_y, occupancy_scale_x)
        LOGGER.info(
            "Loaded mask %s with foreground coverage %.3f%%",
            args.mask_path,
            100.0 * float(mask.mean()),
        )
    else:
        LOGGER.info("No mask supplied. Using the entire zarr.")

    blocks = iter_blocks(
        image_shape=(height, width),
        patch_size=patch_size,
        stride=patch_stride,
        mask_lowres=mask_lowres,
        occupancy_scale=(occupancy_scale_y, occupancy_scale_x),
    )
    LOGGER.info("Selected %d patches for inference.", len(blocks))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OmeZarrBlockDataset(reader=reader, blocks=blocks, patch_size=patch_size)
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
        loader_kwargs["prefetch_factor"] = DEFAULT_PREFETCH_FACTOR
    loader = DataLoader(**loader_kwargs)

    inferer = SlidingWindowInferer(
        roi_size=(roi_size, roi_size),
        sw_batch_size=DEFAULT_SW_BATCH_SIZE,
        overlap=0.0,
        mode="gaussian",
        sigma_scale=0.125,
        padding_mode="replicate",
        sw_device=device,
        device=device,
        cache_roi_weight_map=True,
        progress=False,
    )
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

        if len(dataset) > 0:
            run_block_inference(
                loader=loader,
                model=model,
                inferer=inferer,
                prob_sum_store=prob_sum_store,
                weight_sum_store=weight_sum_store,
                weight_map=weight_map,
                mask=mask,
                device=device,
                amp=True,
            )
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


def infer_folder(args: argparse.Namespace, model: torch.nn.Module, cfg) -> None:
    folder = Path(args.folder)
    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"--folder is not a directory: {folder}")

    checkpoint_stem = Path(args.checkpoint).stem
    direction = "reverse" if bool(args.reverse_layers) else "forward"
    date_str = datetime.now().strftime("%d%m%y")

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

        output_tiff = segment_dir / "preds" / f"{segment_name}_{checkpoint_stem}_{direction}_{date_str}.tif"
        LOGGER.info(
            "Running segment=%s input=%s output=%s",
            segment_name,
            input_zarr,
            output_tiff,
        )
        infer_single_zarr(
            args=args,
            model=model,
            cfg=cfg,
            input_zarr=input_zarr,
            output_tiff=output_tiff,
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

    model, cfg, _metadata = configure_model(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if args.folder is not None:
        infer_folder(args=args, model=model, cfg=cfg)
    else:
        infer_single_zarr(
            args=args,
            model=model,
            cfg=cfg,
            input_zarr=args.input_zarr,
            output_tiff=args.output_tiff,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
