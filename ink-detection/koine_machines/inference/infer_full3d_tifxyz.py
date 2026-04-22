#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import itertools
import logging
import math
import shutil
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence
from urllib.parse import urlparse

import fsspec
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
from numcodecs import Blosc
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from koine_machines.models.load_checkpoint import _load_model_state_with_ddp_compat
from koine_machines.models.make_model import make_model


LOGGER = logging.getLogger("infer_full3d_tifxyz")
PUBLIC_VOLUME_SUBSTRING = "vesuvius-challenge-open-data"
DEFAULT_LEVELS = 6
DEFAULT_OVERLAP = 0.5
DEFAULT_PREFETCH_FACTOR = 2
ARRAY_DIMENSIONS = ["z", "y", "x"]
AXES = [
    {"name": "z", "type": "space"},
    {"name": "y", "type": "space"},
    {"name": "x", "type": "space"},
]
COMPRESSOR = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)


@dataclass(frozen=True, order=True)
class ChunkId:
    z: int
    y: int
    x: int


@dataclass(frozen=True)
class PatchSpec:
    z0: int
    y0: int
    x0: int


@dataclass
class ModelBundle:
    model: torch.nn.Module
    config: dict[str, Any]
    patch_size: tuple[int, int, int]
    target_name: str
    amp_dtype: torch.dtype | None


class TargetHeadWrapper(nn.Module):
    def __init__(self, model: torch.nn.Module, *, target_name: str):
        super().__init__()
        self.model = model
        self.target_name = str(target_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.model(x)
        if not isinstance(outputs, dict):
            raise TypeError(
                f"Expected model to return a dict of target heads, got {type(outputs).__name__}"
            )
        if self.target_name not in outputs:
            raise KeyError(
                f"Model output is missing target {self.target_name!r}; "
                f"available targets: {sorted(outputs.keys())!r}"
            )
        logits = outputs[self.target_name]
        if isinstance(logits, (list, tuple)):
            if len(logits) == 0:
                raise ValueError(f"Target {self.target_name!r} returned an empty output list")
            logits = logits[0]
        return logits


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run full-3D koine_machines checkpoint inference over the zarr chunks "
            "covered by a tifxyz crop and write a sparse six-level uint8 OME-Zarr."
        )
    )
    parser.add_argument("tifxyz_dir", type=Path, help="Folder containing x.tif, y.tif, z.tif, meta.json, and volume_source.txt.")
    parser.add_argument("checkpoint", type=Path, help="koine_machines training checkpoint.")
    parser.add_argument("output_zarr", type=Path, help="Output OME-Zarr directory.")
    parser.add_argument("--resolution", default="0", help="Input zarr pyramid level to infer. Default: 0.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output_zarr if it already exists.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", "--workers", dest="num_workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=DEFAULT_PREFETCH_FACTOR)
    parser.add_argument("--downsample-workers", type=int, default=1)
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, help="Sliding-window overlap fraction. Default: 0.5.")
    parser.add_argument("--chunk-halo", type=int, default=1, help="Extra chunk radius added around tifxyz-occupied chunks. Default: 1.")
    parser.add_argument(
        "--write-region",
        choices=("expanded", "occupied"),
        default="expanded",
        help=(
            "Chunks to write at level 0. 'expanded' writes occupied chunks plus --chunk-halo; "
            "'occupied' writes only chunks containing tifxyz points. Default: expanded."
        ),
    )
    parser.add_argument(
        "--blend-mode",
        choices=("gaussian", "constant"),
        default="gaussian",
        help="How overlapping patch probabilities are blended. Default: gaussian.",
    )
    parser.add_argument("--tta", action="store_true", help="Enable mirror TTA using the original patch plus all 7 non-empty Z/Y/X flips.")
    parser.add_argument("--tta-batch-size", type=int, default=None, help="Maximum TTA variants evaluated per forward pass.")
    parser.add_argument(
        "--amp-dtype",
        choices=("auto", "default", "fp16", "bf16"),
        default="auto",
        help="CUDA autocast dtype. 'auto' reads checkpoint config.mixed_precision.",
    )
    parser.add_argument("--compile-mode", default="reduce-overhead")
    parser.add_argument("--no-compile", dest="compile_model", action="store_false")
    parser.set_defaults(compile_model=True)
    parser.add_argument("--gpus", default=None, help="Comma-separated CUDA ids. Multiple ids use DataParallel.")
    parser.add_argument("--foreground-channel", type=int, default=1, help="Foreground class for C>1 softmax outputs. Default: 1.")
    parser.add_argument("--plan-only", action="store_true", help="Build and print the chunk/patch plan without loading the model or writing output.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if int(args.batch_size) <= 0:
        parser.error("--batch-size must be positive")
    if int(args.num_workers) < 0:
        parser.error("--num-workers must be >= 0")
    if int(args.prefetch_factor) <= 0:
        parser.error("--prefetch-factor must be positive")
    if int(args.downsample_workers) <= 0:
        parser.error("--downsample-workers must be positive")
    if not (0.0 <= float(args.overlap) < 1.0):
        parser.error("--overlap must be in [0, 1)")
    if int(args.chunk_halo) < 0:
        parser.error("--chunk-halo must be >= 0")
    if args.tta_batch_size is not None and int(args.tta_batch_size) <= 0:
        parser.error("--tta-batch-size must be positive")
    args.gpu_ids = parse_gpu_ids(args.gpus)
    return args


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, str(level).upper()),
        format="[%(asctime)s] %(levelname)s %(message)s",
    )


def parse_gpu_ids(value: str | None) -> list[int]:
    if value is None or str(value).strip() == "":
        return []
    gpu_ids: list[int] = []
    seen: set[int] = set()
    for raw in str(value).split(","):
        text = raw.strip()
        if not text:
            raise ValueError("--gpus must be a comma-separated list like 0,1")
        gpu_id = int(text)
        if gpu_id < 0:
            raise ValueError(f"GPU ids must be non-negative, got {gpu_id}")
        if gpu_id in seen:
            raise ValueError(f"Duplicate GPU id in --gpus={value!r}")
        seen.add(gpu_id)
        gpu_ids.append(gpu_id)
    return gpu_ids


def is_url_like(path: str | Path) -> bool:
    parsed = urlparse(str(path))
    return bool(parsed.scheme and parsed.netloc)


def _https_s3_to_s3_url(path: str) -> str | None:
    parsed = urlparse(path)
    if parsed.scheme not in {"http", "https"}:
        return None
    host = parsed.netloc
    suffix = ".s3.us-east-1.amazonaws.com"
    if host == PUBLIC_VOLUME_SUBSTRING:
        bucket = host
    elif host.endswith(suffix):
        bucket = host[: -len(suffix)]
    elif ".s3." in host:
        bucket = host.split(".s3.", 1)[0]
    else:
        return None
    if not bucket:
        return None
    return f"s3://{bucket}{parsed.path}"


def _make_fsspec_store(path: str):
    path_str = str(path).rstrip("/")
    use_anon = PUBLIC_VOLUME_SUBSTRING in path_str
    if use_anon:
        s3_path = path_str if path_str.startswith("s3://") else _https_s3_to_s3_url(path_str)
        if s3_path is not None:
            fs = fsspec.filesystem("s3", anon=True)
            return zarr.storage.FSStore(
                s3_path.rstrip("/"),
                fs=fs,
                mode="r",
                check=False,
                create=False,
                exceptions=(KeyError, FileNotFoundError, PermissionError, OSError),
            )

    if path_str.startswith("s3://"):
        fs = fsspec.filesystem("s3")
        return zarr.storage.FSStore(
            path_str,
            fs=fs,
            mode="r",
            check=False,
            create=False,
            exceptions=(KeyError, FileNotFoundError, PermissionError, OSError),
        )
    if path_str.startswith(("http://", "https://")):
        protocol = "https" if path_str.startswith("https://") else "http"
        fs = fsspec.filesystem(protocol)
        return zarr.storage.FSStore(
            path_str,
            fs=fs,
            mode="r",
            check=False,
            create=False,
            exceptions=(KeyError, FileNotFoundError, PermissionError, OSError),
        )
    return path_str


def open_zarr_root(path: str | Path):
    return zarr.open(_make_fsspec_store(str(path)), mode="r")


def open_zarr_array(path: str | Path, resolution: str):
    root = open_zarr_root(path)
    if isinstance(root, zarr.Array):
        return root
    return root[str(resolution)]


def read_volume_source(tifxyz_dir: Path) -> str:
    source_path = tifxyz_dir / "volume_source.txt"
    if not source_path.is_file():
        raise FileNotFoundError(f"Missing volume_source.txt in {tifxyz_dir}")
    source = source_path.read_text(encoding="utf-8").strip()
    if not source:
        raise ValueError(f"volume_source.txt is empty: {source_path}")
    if not is_url_like(source) and not Path(source).expanduser().is_absolute():
        source = str((tifxyz_dir / source).resolve())
    return source


def read_tifxyz_points(tifxyz_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for name in ("x.tif", "y.tif", "z.tif"):
        path = tifxyz_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing tifxyz coordinate file: {path}")
        arrays.append(np.asarray(tifffile.imread(str(path)), dtype=np.float32))
    x, y, z = arrays
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError(f"x/y/z tif shapes must match, got x={x.shape} y={y.shape} z={z.shape}")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    valid &= (x >= 0.0) & (y >= 0.0) & (z >= 0.0)
    return x, y, z, valid


def chunk_grid_shape(array_shape: Sequence[int], chunk_shape: Sequence[int]) -> tuple[int, int, int]:
    return tuple(
        int(math.ceil(int(shape) / float(int(chunk))))
        for shape, chunk in zip(array_shape, chunk_shape)
    )


def tifxyz_occupied_chunks(
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    valid: np.ndarray,
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
) -> set[ChunkId]:
    if not np.any(valid):
        raise ValueError("No finite non-negative tifxyz points were found.")

    z_idx = np.floor(z[valid]).astype(np.int64, copy=False)
    y_idx = np.floor(y[valid]).astype(np.int64, copy=False)
    x_idx = np.floor(x[valid]).astype(np.int64, copy=False)
    in_bounds = (
        (z_idx >= 0)
        & (z_idx < int(array_shape_zyx[0]))
        & (y_idx >= 0)
        & (y_idx < int(array_shape_zyx[1]))
        & (x_idx >= 0)
        & (x_idx < int(array_shape_zyx[2]))
    )
    if not np.any(in_bounds):
        raise ValueError(
            "No tifxyz points fall inside input volume shape "
            f"{tuple(int(v) for v in array_shape_zyx)}."
        )

    chunk_zyx = np.stack(
        [
            z_idx[in_bounds] // int(chunk_shape_zyx[0]),
            y_idx[in_bounds] // int(chunk_shape_zyx[1]),
            x_idx[in_bounds] // int(chunk_shape_zyx[2]),
        ],
        axis=1,
    )
    unique = np.unique(chunk_zyx, axis=0)
    return {ChunkId(int(row[0]), int(row[1]), int(row[2])) for row in unique}


def expand_chunks(
    chunk_ids: Iterable[ChunkId],
    *,
    radius: int,
    grid_shape_zyx: tuple[int, int, int],
) -> set[ChunkId]:
    radius = int(radius)
    source = set(chunk_ids)
    if radius <= 0:
        return source
    max_z, max_y, max_x = (int(v) - 1 for v in grid_shape_zyx)
    expanded: set[ChunkId] = set()
    for chunk in source:
        for dz in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    zz = int(chunk.z) + dz
                    yy = int(chunk.y) + dy
                    xx = int(chunk.x) + dx
                    if 0 <= zz <= max_z and 0 <= yy <= max_y and 0 <= xx <= max_x:
                        expanded.add(ChunkId(zz, yy, xx))
    return expanded


def chunk_bounds(
    chunk: ChunkId,
    *,
    chunk_shape_zyx: tuple[int, int, int],
    array_shape_zyx: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    starts = (
        int(chunk.z) * int(chunk_shape_zyx[0]),
        int(chunk.y) * int(chunk_shape_zyx[1]),
        int(chunk.x) * int(chunk_shape_zyx[2]),
    )
    stops = tuple(
        min(int(array_shape_zyx[i]), starts[i] + int(chunk_shape_zyx[i]))
        for i in range(3)
    )
    return starts, stops


def chunks_overlapping_box(
    starts: tuple[int, int, int],
    stops: tuple[int, int, int],
    *,
    chunk_shape_zyx: tuple[int, int, int],
    array_shape_zyx: tuple[int, int, int],
) -> Iterator[ChunkId]:
    clipped_starts = tuple(max(0, int(v)) for v in starts)
    clipped_stops = tuple(min(int(array_shape_zyx[i]), int(stops[i])) for i in range(3))
    if any(stop <= start for start, stop in zip(clipped_starts, clipped_stops)):
        return
    ranges = [
        range(
            clipped_starts[i] // int(chunk_shape_zyx[i]),
            ((clipped_stops[i] - 1) // int(chunk_shape_zyx[i])) + 1,
        )
        for i in range(3)
    ]
    for zc, yc, xc in itertools.product(*ranges):
        yield ChunkId(int(zc), int(yc), int(xc))


def _global_patch_starts_for_interval(
    *,
    interval_start: int,
    interval_stop: int,
    volume_length: int,
    patch_size: int,
    stride: int,
) -> list[int]:
    interval_start = int(interval_start)
    interval_stop = int(interval_stop)
    volume_length = int(volume_length)
    patch_size = int(patch_size)
    stride = max(1, int(stride))
    max_start = max(0, volume_length - patch_size)
    first = max(0, ((interval_start - patch_size + 1) // stride) * stride)
    starts = []
    start = first
    while start <= max_start and start < interval_stop:
        if start + patch_size > interval_start:
            starts.append(int(start))
        start += stride
    if interval_stop > max_start and max_start + patch_size > interval_start:
        starts.append(int(max_start))
    return sorted(set(starts))


def build_patch_plan_for_chunks(
    *,
    target_chunks: Iterable[ChunkId],
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    patch_size_zyx: tuple[int, int, int],
    stride_zyx: tuple[int, int, int],
) -> list[PatchSpec]:
    patches: set[PatchSpec] = set()
    for chunk in target_chunks:
        starts, stops = chunk_bounds(
            chunk,
            chunk_shape_zyx=chunk_shape_zyx,
            array_shape_zyx=array_shape_zyx,
        )
        per_axis_starts = [
            _global_patch_starts_for_interval(
                interval_start=starts[axis],
                interval_stop=stops[axis],
                volume_length=array_shape_zyx[axis],
                patch_size=patch_size_zyx[axis],
                stride=stride_zyx[axis],
            )
            for axis in range(3)
        ]
        for z0, y0, x0 in itertools.product(*per_axis_starts):
            patches.add(PatchSpec(int(z0), int(y0), int(x0)))
    return sorted(patches, key=lambda p: (p.z0, p.y0, p.x0))


def compute_chunk_contribution_counts(
    patches: Sequence[PatchSpec],
    *,
    target_chunks: set[ChunkId],
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    patch_size_zyx: tuple[int, int, int],
) -> dict[ChunkId, int]:
    counts = {chunk: 0 for chunk in target_chunks}
    for patch in patches:
        starts = (int(patch.z0), int(patch.y0), int(patch.x0))
        stops = tuple(starts[i] + int(patch_size_zyx[i]) for i in range(3))
        for chunk in chunks_overlapping_box(
            starts,
            stops,
            chunk_shape_zyx=chunk_shape_zyx,
            array_shape_zyx=array_shape_zyx,
        ):
            if chunk in target_chunks:
                counts[chunk] += 1
    return {chunk: count for chunk, count in counts.items() if count > 0}


def _read_bbox_with_padding(array, bbox: tuple[int, int, int, int, int, int], *, fill_value=0) -> tuple[np.ndarray, tuple[slice, slice, slice] | None]:
    z0, y0, x0, z1, y1, x1 = (int(v) for v in bbox)
    expected_shape = (z1 - z0, y1 - y0, x1 - x0)
    if any(size <= 0 for size in expected_shape):
        raise ValueError(f"bbox must define a positive crop, got {bbox!r}")

    src_starts = (max(0, z0), max(0, y0), max(0, x0))
    src_stops = (
        min(int(array.shape[0]), z1),
        min(int(array.shape[1]), y1),
        min(int(array.shape[2]), x1),
    )
    output = np.full(expected_shape, fill_value, dtype=np.dtype(array.dtype))
    if any(stop <= start for start, stop in zip(src_starts, src_stops)):
        return output, None

    crop = np.asarray(
        array[
            src_starts[0] : src_stops[0],
            src_starts[1] : src_stops[1],
            src_starts[2] : src_stops[2],
        ]
    )
    dst_starts = tuple(src_starts[i] - (z0, y0, x0)[i] for i in range(3))
    dst_stops = tuple(dst_starts[i] + crop.shape[i] for i in range(3))
    dst_slices = tuple(slice(dst_starts[i], dst_stops[i]) for i in range(3))
    output[dst_slices] = crop
    return output, dst_slices


def normalize_image_crop(image: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    from koine_machines.data.ink_dataset import _normalize_image_crop

    return _normalize_image_crop(image, config)


class Full3DPatchDataset(Dataset):
    def __init__(
        self,
        *,
        volume_path: str,
        resolution: str,
        patches: Sequence[PatchSpec],
        patch_size_zyx: tuple[int, int, int],
        config: dict[str, Any],
    ):
        self.volume_path = str(volume_path)
        self.resolution = str(resolution)
        self.patches = list(patches)
        self.patch_size_zyx = tuple(int(v) for v in patch_size_zyx)
        self.config = copy.deepcopy(config)
        self._array = None

    def __len__(self) -> int:
        return len(self.patches)

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_array"] = None
        return state

    def _ensure_array(self):
        if self._array is None:
            self._array = open_zarr_array(self.volume_path, self.resolution)
        return self._array

    def __getitem__(self, index: int):
        patch = self.patches[int(index)]
        z0, y0, x0 = int(patch.z0), int(patch.y0), int(patch.x0)
        dz, dy, dx = self.patch_size_zyx
        crop, valid_slices = _read_bbox_with_padding(
            self._ensure_array(),
            (z0, y0, x0, z0 + dz, y0 + dy, x0 + dx),
            fill_value=0,
        )
        crop = crop.astype(np.float32, copy=False)
        if valid_slices is not None:
            crop[valid_slices] = normalize_image_crop(crop[valid_slices], self.config)
        image = torch.from_numpy(np.ascontiguousarray(crop)).float().unsqueeze(0)
        meta = torch.tensor([z0, y0, x0], dtype=torch.int64)
        return image, meta


def load_checkpoint_payload(checkpoint_path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}: expected dict.")
    return payload


def normalize_training_config_for_full3d(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    model_type = str(normalized.get("model_type", "")).strip().lower()
    if model_type == "dinov2":
        model_config = normalized.setdefault("model_config", {})
        for key in ("pretrained_backbone", "pretrained_decoder_type"):
            if key in normalized:
                model_config.setdefault(key, normalized[key])
        normalized["model_type"] = "vesuvius_unet"

    mode = str(normalized.get("mode", "flat")).strip().lower()
    if mode not in {"full_3d", "full_3d_single_wrap"}:
        raise ValueError(f"This script expects mode='full_3d' or 'full_3d_single_wrap', got {mode!r}")

    if mode == "full_3d_single_wrap":
        normalized["in_channels"] = 2
    else:
        normalized.setdefault("in_channels", 1)

    patch_size = normalized.get("crop_size", normalized.get("patch_size"))
    if not isinstance(patch_size, (list, tuple)) or len(patch_size) != 3:
        raise ValueError(
            "Checkpoint config must define crop_size or patch_size as [z, y, x], "
            f"got {patch_size!r}"
        )
    normalized["crop_size"] = [int(v) for v in patch_size]
    normalized["patch_size"] = [int(v) for v in patch_size]
    normalized.setdefault("model_config", {})
    targets = normalized.setdefault("targets", {})
    if not targets:
        targets["ink"] = {"out_channels": 1, "activation": "none", "z_projection_mode": "none"}
    for target_cfg in targets.values():
        if isinstance(target_cfg, dict):
            target_cfg["activation"] = "none"
            target_cfg["z_projection_mode"] = "none"
            if isinstance(target_cfg.get("z_projection"), dict):
                target_cfg["z_projection"]["mode"] = "none"
    normalized.setdefault("use_stitched_forward", False)
    normalized["stitch_factor"] = 1
    return normalized


def infer_target_name_from_config(config: dict[str, Any]) -> str:
    targets = config.get("targets")
    if isinstance(targets, dict) and targets:
        if "ink" in targets:
            return "ink"
        return str(next(iter(targets)))
    return "ink"


def checkpoint_amp_dtype(payload: dict[str, Any]) -> torch.dtype | None:
    config = payload.get("config")
    if not isinstance(config, dict):
        return None
    mixed_precision = str(config.get("mixed_precision", "")).strip().lower()
    if mixed_precision in {"fp16", "float16", "half"}:
        return torch.float16
    if mixed_precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None


def resolve_amp_dtype(value: str, payload: dict[str, Any]) -> torch.dtype | None:
    value = str(value).strip().lower()
    if value == "default":
        return None
    if value == "fp16":
        return torch.float16
    if value == "bf16":
        return torch.bfloat16
    if value == "auto":
        return checkpoint_amp_dtype(payload)
    raise ValueError(f"Unsupported amp dtype {value!r}")


def build_model_bundle(checkpoint_path: Path, *, amp_dtype_arg: str) -> ModelBundle:
    payload = load_checkpoint_payload(checkpoint_path)
    config = payload.get("config")
    if not isinstance(config, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is missing dict entry 'config'.")
    config = normalize_training_config_for_full3d(config)
    model = make_model(config)

    state = payload.get("ema_model")
    selected = "ema_model"
    if not isinstance(state, dict):
        state = payload.get("model")
        selected = "model"
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is missing model weights.")
    _load_model_state_with_ddp_compat(model, state)
    target_name = infer_target_name_from_config(config)
    LOGGER.info("Loaded %s weights from %s for target=%s", selected, checkpoint_path, target_name)
    return ModelBundle(
        model=TargetHeadWrapper(model.eval(), target_name=target_name),
        config=config,
        patch_size=tuple(int(v) for v in config["patch_size"]),
        target_name=target_name,
        amp_dtype=resolve_amp_dtype(amp_dtype_arg, payload),
    )


def prepare_model(
    bundle: ModelBundle,
    *,
    gpu_ids: Sequence[int],
    compile_model: bool,
    compile_mode: str,
) -> tuple[ModelBundle, torch.device, bool]:
    gpu_ids = tuple(int(v) for v in gpu_ids)
    if gpu_ids:
        if not torch.cuda.is_available():
            raise ValueError("--gpus was provided, but CUDA is not available.")
        count = int(torch.cuda.device_count())
        invalid = [gpu_id for gpu_id in gpu_ids if gpu_id >= count]
        if invalid:
            raise ValueError(f"Requested unavailable CUDA ids {invalid}; visible CUDA device count is {count}.")
        device = torch.device(f"cuda:{gpu_ids[0]}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = bundle.model.to(device)
    compile_enabled = bool(compile_model)
    if len(gpu_ids) > 1:
        if compile_enabled:
            LOGGER.warning("Disabling torch.compile because DataParallel is enabled for --gpus=%s", ",".join(map(str, gpu_ids)))
            compile_enabled = False
        model = nn.DataParallel(model, device_ids=list(gpu_ids), output_device=gpu_ids[0])

    if compile_enabled:
        compile_fn = getattr(torch, "compile", None)
        if compile_fn is None:
            LOGGER.warning("torch.compile is unavailable; continuing without compilation.")
            compile_enabled = False
        else:
            try:
                model = compile_fn(model, mode=str(compile_mode), fullgraph=False, dynamic=False)
            except Exception as exc:
                LOGGER.warning("torch.compile failed (%s); continuing without compilation.", exc)
                compile_enabled = False

    bundle.model = model
    return bundle, device, compile_enabled


def create_importance_map(
    patch_size_zyx: tuple[int, int, int],
    *,
    mode: str,
    sigma_scale: float = 0.125,
) -> np.ndarray:
    if mode == "constant":
        return np.ones(patch_size_zyx, dtype=np.float32)
    if mode != "gaussian":
        raise ValueError(f"Unsupported blend mode {mode!r}")
    axes = []
    for size in patch_size_zyx:
        size = int(size)
        sigma = max(float(size) * float(sigma_scale), 1e-6)
        coord = np.arange(size, dtype=np.float32) - ((size - 1) / 2.0)
        axes.append(np.exp(-0.5 * (coord / sigma) ** 2).astype(np.float32))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    weight /= max(float(weight.max()), np.finfo(np.float32).eps)
    return np.clip(weight, np.finfo(np.float32).eps, None).astype(np.float32, copy=False)


def tta_variants(enabled: bool) -> list[tuple[int, ...]]:
    if not enabled:
        return [()]
    axes = (0, 1, 2)
    return [
        combo
        for count in range(0, len(axes) + 1)
        for combo in itertools.combinations(axes, count)
    ]


def flip_spatial(tensor: torch.Tensor, axes: Sequence[int]) -> torch.Tensor:
    if not axes:
        return tensor
    return torch.flip(tensor, dims=[int(axis) + 2 for axis in axes])


def logits_to_probabilities(
    logits: torch.Tensor,
    *,
    patch_size_zyx: tuple[int, int, int],
    foreground_channel: int,
) -> torch.Tensor:
    if logits.ndim != 5:
        raise ValueError(f"Expected logits [B,C,Z,Y,X], got shape={tuple(logits.shape)}")
    if tuple(int(v) for v in logits.shape[-3:]) != tuple(int(v) for v in patch_size_zyx):
        logits = F.interpolate(
            logits.float(),
            size=tuple(int(v) for v in patch_size_zyx),
            mode="trilinear",
            align_corners=False,
        )
    channels = int(logits.shape[1])
    if channels == 1:
        return logits.float().sigmoid()
    if channels == 2:
        return logits.float().softmax(dim=1)[:, 1:2]
    foreground_channel = int(foreground_channel)
    if not 0 <= foreground_channel < channels:
        raise ValueError(f"foreground_channel={foreground_channel} is invalid for {channels} output channels")
    return logits.float().softmax(dim=1)[:, foreground_channel : foreground_channel + 1]


def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    variants: Sequence[tuple[int, ...]],
    tta_batch_size: int | None,
    patch_size_zyx: tuple[int, int, int],
    foreground_channel: int,
) -> torch.Tensor:
    variants = list(variants)
    if len(variants) == 1:
        return logits_to_probabilities(
            model(images),
            patch_size_zyx=patch_size_zyx,
            foreground_channel=foreground_channel,
        )

    batch_size = int(images.shape[0])
    max_variants = len(variants) if tta_batch_size is None else min(int(tta_batch_size), len(variants))
    prob_sum = None
    for start in range(0, len(variants), max_variants):
        chunk = variants[start : start + max_variants]
        augmented = torch.cat([flip_spatial(images, axes) for axes in chunk], dim=0)
        probs = logits_to_probabilities(
            model(augmented),
            patch_size_zyx=patch_size_zyx,
            foreground_channel=foreground_channel,
        )
        probs = probs.reshape(len(chunk), batch_size, *probs.shape[1:])
        for variant_index, axes in enumerate(chunk):
            restored = flip_spatial(probs[variant_index], axes)
            prob_sum = restored if prob_sum is None else prob_sum + restored
    assert prob_sum is not None
    return prob_sum / float(len(variants))


class ChunkAccumulator3D:
    def __init__(
        self,
        *,
        output: zarr.Array,
        target_chunks: set[ChunkId],
        contribution_counts: dict[ChunkId, int],
        chunk_shape_zyx: tuple[int, int, int],
    ):
        self.output = output
        self.target_chunks = set(target_chunks)
        self.contribution_counts = dict(contribution_counts)
        self.chunk_shape_zyx = tuple(int(v) for v in chunk_shape_zyx)
        self.array_shape_zyx = tuple(int(v) for v in output.shape)
        self.seen_counts: dict[ChunkId, int] = {}
        self.buffers: dict[ChunkId, tuple[np.ndarray, np.ndarray]] = {}

    def _bounds(self, chunk: ChunkId) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        return chunk_bounds(
            chunk,
            chunk_shape_zyx=self.chunk_shape_zyx,
            array_shape_zyx=self.array_shape_zyx,
        )

    def _get_buffers(self, chunk: ChunkId) -> tuple[np.ndarray, np.ndarray]:
        buffers = self.buffers.get(chunk)
        if buffers is not None:
            return buffers
        starts, stops = self._bounds(chunk)
        shape = tuple(stops[i] - starts[i] for i in range(3))
        buffers = (
            np.zeros(shape, dtype=np.float32),
            np.zeros(shape, dtype=np.float32),
        )
        self.buffers[chunk] = buffers
        return buffers

    def add_patch(
        self,
        *,
        patch_start_zyx: tuple[int, int, int],
        probabilities: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        patch_start_zyx = tuple(int(v) for v in patch_start_zyx)
        patch_stop_zyx = tuple(patch_start_zyx[i] + int(probabilities.shape[i]) for i in range(3))
        for chunk in chunks_overlapping_box(
            patch_start_zyx,
            patch_stop_zyx,
            chunk_shape_zyx=self.chunk_shape_zyx,
            array_shape_zyx=self.array_shape_zyx,
        ):
            if chunk not in self.target_chunks:
                continue
            chunk_start, chunk_stop = self._bounds(chunk)
            inter_start = tuple(max(patch_start_zyx[i], chunk_start[i]) for i in range(3))
            inter_stop = tuple(min(patch_stop_zyx[i], chunk_stop[i]) for i in range(3))
            if any(stop <= start for start, stop in zip(inter_start, inter_stop)):
                continue

            patch_slices = tuple(
                slice(inter_start[i] - patch_start_zyx[i], inter_stop[i] - patch_start_zyx[i])
                for i in range(3)
            )
            chunk_slices = tuple(
                slice(inter_start[i] - chunk_start[i], inter_stop[i] - chunk_start[i])
                for i in range(3)
            )
            prob_buffer, weight_buffer = self._get_buffers(chunk)
            weight_view = weights[patch_slices]
            prob_buffer[chunk_slices] += probabilities[patch_slices] * weight_view
            weight_buffer[chunk_slices] += weight_view

            seen = self.seen_counts.get(chunk, 0) + 1
            expected = self.contribution_counts[chunk]
            if seen >= expected:
                self._flush(chunk)
            else:
                self.seen_counts[chunk] = seen

    def _flush(self, chunk: ChunkId) -> None:
        prob_buffer, weight_buffer = self.buffers.pop(chunk)
        self.seen_counts.pop(chunk, None)
        np.divide(prob_buffer, weight_buffer, out=prob_buffer, where=weight_buffer > 1e-6)
        np.clip(prob_buffer, 0.0, 1.0, out=prob_buffer)
        out = np.rint(prob_buffer * 255.0).astype(np.uint8, copy=False)
        starts, stops = self._bounds(chunk)
        self.output[
            starts[0] : stops[0],
            starts[1] : stops[1],
            starts[2] : stops[2],
        ] = out

    def flush_remaining(self) -> None:
        for chunk in list(self.buffers):
            self._flush(chunk)


def multiscales_metadata(name: str, levels: int) -> dict[str, Any]:
    datasets = []
    for level in range(levels):
        scale = float(2**level)
        datasets.append(
            {
                "path": str(level),
                "coordinateTransformations": [{"type": "scale", "scale": [scale, scale, scale]}],
            }
        )
    return {"multiscales": [{"name": name, "version": "0.4", "axes": AXES, "datasets": datasets}]}


def pyramid_shapes(shape_zyx: tuple[int, int, int], levels: int) -> list[tuple[int, int, int]]:
    shapes = []
    current = tuple(int(v) for v in shape_zyx)
    for _ in range(int(levels)):
        shapes.append(current)
        current = tuple((int(v) + 1) // 2 for v in current)
    return shapes


def create_output_zarr(
    output_path: Path,
    *,
    shape_zyx: tuple[int, int, int],
    chunks_zyx: tuple[int, int, int],
    levels: int,
    overwrite: bool,
) -> list[zarr.Array]:
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")
        shutil.rmtree(output_path)
    group = zarr.open_group(str(output_path), mode="w")
    group.attrs.update(multiscales_metadata(output_path.stem, int(levels)))
    arrays = []
    for level, shape in enumerate(pyramid_shapes(shape_zyx, levels)):
        chunks = tuple(max(1, min(int(chunks_zyx[i]), int(shape[i]))) for i in range(3))
        arr = group.create_dataset(
            str(level),
            shape=shape,
            chunks=chunks,
            dtype=np.uint8,
            compressor=COMPRESSOR,
            fill_value=0,
            overwrite=True,
            dimension_separator="/",
            write_empty_chunks=False,
        )
        arr.attrs["_ARRAY_DIMENSIONS"] = ARRAY_DIMENSIONS
        arrays.append(arr)
    return arrays


def downsample_mean_3d(block: np.ndarray) -> np.ndarray:
    block = np.asarray(block)
    out_shape = tuple((int(v) + 1) // 2 for v in block.shape)
    accum = np.zeros(out_shape, dtype=np.float64)
    counts = np.zeros(out_shape, dtype=np.float64)
    for dz in (0, 1):
        for dy in (0, 1):
            for dx in (0, 1):
                sub = block[dz::2, dy::2, dx::2]
                if sub.size == 0:
                    continue
                slices = tuple(slice(0, int(v)) for v in sub.shape)
                accum[slices] += sub
                counts[slices] += 1.0
    out = np.rint(accum / counts).astype(block.dtype, copy=False)
    return np.ascontiguousarray(out)


def scale_chunks_to_next_level(
    chunks: Iterable[ChunkId],
    *,
    source_array: zarr.Array,
    target_array: zarr.Array,
) -> set[ChunkId]:
    target_chunks: set[ChunkId] = set()
    source_shape = tuple(int(v) for v in source_array.shape)
    source_chunks = tuple(int(v) for v in source_array.chunks)
    target_shape = tuple(int(v) for v in target_array.shape)
    target_chunk_shape = tuple(int(v) for v in target_array.chunks)
    for chunk in chunks:
        starts, stops = chunk_bounds(
            chunk,
            chunk_shape_zyx=source_chunks,
            array_shape_zyx=source_shape,
        )
        target_starts = tuple(starts[i] // 2 for i in range(3))
        target_stops = tuple((stops[i] + 1) // 2 for i in range(3))
        target_chunks.update(
            chunks_overlapping_box(
                target_starts,
                target_stops,
                chunk_shape_zyx=target_chunk_shape,
                array_shape_zyx=target_shape,
            )
        )
    return target_chunks


def downsample_level_chunk(source: zarr.Array, target: zarr.Array, chunk: ChunkId) -> None:
    target_shape = tuple(int(v) for v in target.shape)
    target_chunks = tuple(int(v) for v in target.chunks)
    target_starts, target_stops = chunk_bounds(
        chunk,
        chunk_shape_zyx=target_chunks,
        array_shape_zyx=target_shape,
    )
    source_starts = tuple(int(v) * 2 for v in target_starts)
    source_stops = tuple(min(int(source.shape[i]), int(target_stops[i]) * 2) for i in range(3))
    source_block = np.asarray(
        source[
            source_starts[0] : source_stops[0],
            source_starts[1] : source_stops[1],
            source_starts[2] : source_stops[2],
        ]
    )
    downsampled = downsample_mean_3d(source_block)
    target[
        target_starts[0] : target_starts[0] + downsampled.shape[0],
        target_starts[1] : target_starts[1] + downsampled.shape[1],
        target_starts[2] : target_starts[2] + downsampled.shape[2],
    ] = downsampled


def build_downsample_levels(
    arrays: Sequence[zarr.Array],
    *,
    level0_written_chunks: set[ChunkId],
    workers: int,
) -> None:
    current_chunks = set(level0_written_chunks)
    for level in range(1, len(arrays)):
        source = arrays[level - 1]
        target = arrays[level]
        target_chunks = scale_chunks_to_next_level(
            current_chunks,
            source_array=source,
            target_array=target,
        )
        desc = f"Downsample L{level}"
        ordered = sorted(target_chunks)
        if workers <= 1 or len(ordered) <= 1:
            for chunk in tqdm(ordered, desc=desc, unit="chunk"):
                downsample_level_chunk(source, target, chunk)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            with ThreadPoolExecutor(max_workers=min(int(workers), len(ordered))) as executor:
                futures = [executor.submit(downsample_level_chunk, source, target, chunk) for chunk in ordered]
                for future in tqdm(as_completed(futures), total=len(futures), desc=desc, unit="chunk"):
                    future.result()
        LOGGER.info("Built level %d for %d chunks", level, len(ordered))
        current_chunks = target_chunks


def run_inference(
    *,
    loader: DataLoader,
    bundle: ModelBundle,
    device: torch.device,
    accumulator: ChunkAccumulator3D,
    weight_map: np.ndarray,
    tta_enabled: bool,
    tta_batch_size: int | None,
    foreground_channel: int,
) -> None:
    variants = tta_variants(tta_enabled)
    LOGGER.info("TTA variants=%d%s", len(variants), " (7 flips + original)" if tta_enabled else "")
    autocast_enabled = device.type == "cuda"
    if autocast_enabled and bundle.amp_dtype is not None:
        LOGGER.info("CUDA autocast dtype=%s", str(bundle.amp_dtype).replace("torch.", ""))
    amp_context = (
        torch.autocast(device_type="cuda", enabled=True, dtype=bundle.amp_dtype)
        if autocast_enabled
        else nullcontext()
    )
    with torch.inference_mode(), amp_context:
        for images, meta in tqdm(loader, desc="Infer", unit="patch"):
            images = images.to(device, non_blocking=True)
            probs = predict_batch(
                bundle.model,
                images,
                variants=variants,
                tta_batch_size=tta_batch_size,
                patch_size_zyx=bundle.patch_size,
                foreground_channel=foreground_channel,
            )
            probs_np = probs[:, 0].cpu().numpy()
            meta_np = meta.cpu().numpy()
            for i in range(probs_np.shape[0]):
                z0, y0, x0 = (int(v) for v in meta_np[i])
                accumulator.add_patch(
                    patch_start_zyx=(z0, y0, x0),
                    probabilities=probs_np[i],
                    weights=weight_map,
                )
    accumulator.flush_remaining()


def summarize_plan(
    *,
    volume_path: str,
    array_shape_zyx: tuple[int, int, int],
    chunk_shape_zyx: tuple[int, int, int],
    occupied_chunks: set[ChunkId],
    expanded_chunks: set[ChunkId],
    target_chunks: set[ChunkId],
    patches: Sequence[PatchSpec],
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
) -> None:
    LOGGER.info("Input volume: %s", volume_path)
    LOGGER.info("Input shape=%s chunks=%s", array_shape_zyx, chunk_shape_zyx)
    LOGGER.info(
        "Patch size=%s stride=%s occupied_chunks=%d expanded_chunks=%d target_chunks=%d inference_patches=%d",
        patch_size,
        stride,
        len(occupied_chunks),
        len(expanded_chunks),
        len(target_chunks),
        len(patches),
    )
    if target_chunks:
        z_vals = [chunk.z for chunk in target_chunks]
        y_vals = [chunk.y for chunk in target_chunks]
        x_vals = [chunk.x for chunk in target_chunks]
        LOGGER.info(
            "Target chunk ranges z=[%d,%d] y=[%d,%d] x=[%d,%d]",
            min(z_vals),
            max(z_vals),
            min(y_vals),
            max(y_vals),
            min(x_vals),
            max(x_vals),
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    tifxyz_dir = Path(args.tifxyz_dir)
    volume_path = read_volume_source(tifxyz_dir)
    volume = open_zarr_array(volume_path, str(args.resolution))
    array_shape = tuple(int(v) for v in volume.shape)
    chunk_shape = tuple(int(v) for v in (volume.chunks or volume.shape))
    if len(array_shape) != 3:
        raise ValueError(f"Expected a 3D ZYX input array, got shape={array_shape}")
    if len(chunk_shape) != 3:
        raise ValueError(f"Expected 3D zarr chunks, got chunks={chunk_shape}")

    x, y, z, valid = read_tifxyz_points(tifxyz_dir)
    occupied_chunks = tifxyz_occupied_chunks(
        x=x,
        y=y,
        z=z,
        valid=valid,
        array_shape_zyx=array_shape,
        chunk_shape_zyx=chunk_shape,
    )
    expanded_chunks = expand_chunks(
        occupied_chunks,
        radius=int(args.chunk_halo),
        grid_shape_zyx=chunk_grid_shape(array_shape, chunk_shape),
    )
    target_chunks = expanded_chunks if args.write_region == "expanded" else occupied_chunks

    if args.plan_only:
        payload = load_checkpoint_payload(args.checkpoint)
        config = normalize_training_config_for_full3d(payload["config"])
        patch_size = tuple(int(v) for v in config["patch_size"])
    else:
        bundle = build_model_bundle(args.checkpoint, amp_dtype_arg=args.amp_dtype)
        patch_size = bundle.patch_size

    stride = tuple(max(1, int(round(int(v) * (1.0 - float(args.overlap))))) for v in patch_size)
    patches = build_patch_plan_for_chunks(
        target_chunks=target_chunks,
        array_shape_zyx=array_shape,
        chunk_shape_zyx=chunk_shape,
        patch_size_zyx=patch_size,
        stride_zyx=stride,
    )
    contribution_counts = compute_chunk_contribution_counts(
        patches,
        target_chunks=set(target_chunks),
        array_shape_zyx=array_shape,
        chunk_shape_zyx=chunk_shape,
        patch_size_zyx=patch_size,
    )
    missing = set(target_chunks) - set(contribution_counts)
    if missing:
        raise RuntimeError(f"Patch plan failed to cover {len(missing)} target chunks.")

    summarize_plan(
        volume_path=volume_path,
        array_shape_zyx=array_shape,
        chunk_shape_zyx=chunk_shape,
        occupied_chunks=occupied_chunks,
        expanded_chunks=expanded_chunks,
        target_chunks=set(target_chunks),
        patches=patches,
        patch_size=patch_size,
        stride=stride,
    )
    if args.plan_only:
        return 0

    bundle, device, compile_enabled = prepare_model(
        bundle,
        gpu_ids=args.gpu_ids,
        compile_model=bool(args.compile_model),
        compile_mode=str(args.compile_mode),
    )
    LOGGER.info(
        "Using device=%s compile=%s batch_size=%d num_workers=%d target=%s output=%s",
        device,
        compile_enabled,
        int(args.batch_size),
        int(args.num_workers),
        bundle.target_name,
        args.output_zarr,
    )

    arrays = create_output_zarr(
        Path(args.output_zarr),
        shape_zyx=array_shape,
        chunks_zyx=chunk_shape,
        levels=DEFAULT_LEVELS,
        overwrite=bool(args.overwrite),
    )
    dataset = Full3DPatchDataset(
        volume_path=volume_path,
        resolution=str(args.resolution),
        patches=patches,
        patch_size_zyx=patch_size,
        config=bundle.config,
    )
    effective_batch_size = int(args.batch_size) * max(1, len(args.gpu_ids))
    loader_kwargs = {
        "dataset": dataset,
        "batch_size": effective_batch_size,
        "shuffle": False,
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "drop_last": False,
    }
    if int(args.num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    loader = DataLoader(**loader_kwargs)
    weights = create_importance_map(patch_size, mode=str(args.blend_mode))
    accumulator = ChunkAccumulator3D(
        output=arrays[0],
        target_chunks=set(target_chunks),
        contribution_counts=contribution_counts,
        chunk_shape_zyx=chunk_shape,
    )
    run_inference(
        loader=loader,
        bundle=bundle,
        device=device,
        accumulator=accumulator,
        weight_map=weights,
        tta_enabled=bool(args.tta),
        tta_batch_size=args.tta_batch_size,
        foreground_channel=int(args.foreground_channel),
    )
    LOGGER.info("Wrote level 0 chunks: %d", len(target_chunks))
    build_downsample_levels(
        arrays,
        level0_written_chunks=set(target_chunks),
        workers=int(args.downsample_workers),
    )
    LOGGER.info("Wrote OME-Zarr: %s", args.output_zarr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
