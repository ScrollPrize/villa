"""Resize task for zarr arrays.

Resizes an OME-Zarr to match the shape of a reference OME-Zarr using cv2 resize.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import zarr

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task


@dataclass
class ResizeConfig(TaskConfig):
    """Configuration for resize task."""

    reference_zarr: str = ""
    interpolation: str = "linear"
    output_dtype: Optional[np.dtype] = None


def _resize_chunk_3d(
    data: np.ndarray, target_shape: Tuple[int, int, int], interpolation: int
) -> np.ndarray:
    """Resize a 3D chunk by resizing each z-slice with cv2."""
    target_z, target_y, target_x = target_shape

    # First resize in YX for each z-slice
    resized_yx = np.zeros((data.shape[0], target_y, target_x), dtype=data.dtype)
    for z in range(data.shape[0]):
        resized_yx[z] = cv2.resize(
            data[z],
            (target_x, target_y),  # cv2 uses (width, height)
            interpolation=interpolation,
        )

    # Then resize in Z if needed
    if data.shape[0] != target_z:
        # Resize along z axis by resizing each x-column
        resized = np.zeros((target_z, target_y, target_x), dtype=data.dtype)
        for y in range(target_y):
            # Treat the ZX plane as an image and resize
            zx_plane = resized_yx[:, y, :]  # shape: (z, x)
            resized_zx = cv2.resize(
                zx_plane,
                (target_x, target_z),  # cv2 uses (width, height) -> (x, z)
                interpolation=interpolation,
            )
            resized[:, y, :] = resized_zx
        return resized

    return resized_yx


# Input voxels of context needed on each side of an output chunk; covers the
# widest cv2 kernel (Lanczos4: 3 taps left, 4 right) plus one for rounding.
_HALO = 5
# Largest reduced-ratio input block for which the exact cv2 path is used.
_MAX_EXACT_BLOCK = 64
# cv2's cubic/Lanczos SIMD paths round differently depending on image width, so
# a cropped window is not bit-identical to the whole plane; those modes always
# use the tap path.
_EXACT_MODES = (cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA)


def _axis_block(n_in: int, n_out: int) -> Tuple[int, int]:
    """Reduced ratio (input block, output block) with n_in/n_out == N/M."""
    g = int(np.gcd(n_in, n_out))
    return n_in // g, n_out // g


def _aligned_window(o0: int, o1: int, n_in: int, n_out: int) -> Tuple[int, int, int, int]:
    """Output/input window aligned to the reduced ratio, padded by a halo.

    Returns (wo0, wo1, wi0, wi1). Because both windows start and end on
    multiples of the reduced ratio, cv2.resize on the input window computes
    exactly the same scale and per-pixel source coordinates (shifted by an
    integer) as a whole-axis resize.
    """
    N, M = _axis_block(n_in, n_out)
    pad_blocks = -(-_HALO // N)
    b0 = max(o0 // M - pad_blocks, 0)
    b1 = min(-(-o1 // M) + pad_blocks, n_out // M)
    return b0 * M, b1 * M, b0 * N, b1 * N


def _resize_chunk_exact(input_data, out_start, out_end, input_shape, output_shape,
                        interpolation):
    windows = [
        _aligned_window(o0, o1, ni, no)
        for o0, o1, ni, no in zip(out_start, out_end, input_shape, output_shape)
    ]
    src = input_data[tuple(slice(w[2], w[3]) for w in windows)]
    if not src.any():
        return None
    win_shape = tuple(w[1] - w[0] for w in windows)
    resized = _resize_chunk_3d(np.ascontiguousarray(src), win_shape, interpolation)
    return resized[tuple(slice(o0 - w[0], o1 - w[0])
                         for o0, o1, w in zip(out_start, out_end, windows))]


def _cubic_w(t):
    A = -0.75
    return np.stack([
        ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A,
        ((A + 2) * t - (A + 3)) * t * t + 1,
        ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1,
        1 - (((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A)
        - (((A + 2) * t - (A + 3)) * t * t + 1)
        - (((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1),
    ], axis=-1)


def _lanczos_w(t):
    k = np.arange(-3, 5)
    x = (t[:, None] - k[None, :]).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = np.where(np.abs(x) < 1e-12, 1.0,
                     4 * np.sin(np.pi * x) * np.sin(np.pi * x / 4) / (np.pi * np.pi * x * x))
    return w / w.sum(axis=1, keepdims=True)


def _axis_taps(o0: int, o1: int, n_in: int, n_out: int, interpolation: int,
               true_area: bool = True):
    """Global (cv2-convention) source indices and weights for outputs o0..o1.

    For INTER_AREA, cv2 only box-filters when *both* axes of the 2D call shrink
    (or stay equal); otherwise both axes use its upscale-style coefficients.
    ``true_area`` carries that 2D decision.
    """
    inv = n_out / n_in
    scale = 1.0 / inv
    d = np.arange(o0, o1, dtype=np.float64)
    if n_in == n_out:
        return np.arange(o0, o1)[:, None], np.ones((o1 - o0, 1))
    if interpolation == cv2.INTER_NEAREST:
        idx = np.minimum(np.floor(d * scale).astype(np.int64), n_in - 1)
        return idx[:, None], np.ones((o1 - o0, 1))
    if interpolation == cv2.INTER_AREA and true_area:
        lo = d * scale
        hi = lo + scale
        first = np.floor(lo).astype(np.int64)
        ntap = int(np.ceil(scale)) + 1
        idx = first[:, None] + np.arange(ntap)[None, :]
        w = np.clip(np.minimum(hi[:, None], idx + 1) - np.maximum(lo[:, None], idx), 0, None)
        w /= scale
        return np.clip(idx, 0, n_in - 1), w
    if interpolation == cv2.INTER_AREA:
        sx = np.floor(d * scale).astype(np.int64)
        fx = (d + 1) - (sx + 1) * inv
        fx = np.where(fx <= 0, 0.0, fx - np.floor(fx))
        fx = np.where(sx >= n_in - 1, 0.0, fx)
        sx = np.minimum(sx, n_in - 1)
        idx = np.stack([sx, np.minimum(sx + 1, n_in - 1)], axis=1)
        return idx, np.stack([1 - fx, fx], axis=1)
    fx = (d + 0.5) * scale - 0.5
    sx = np.floor(fx).astype(np.int64)
    t = fx - sx
    if interpolation == cv2.INTER_LINEAR:
        t = np.where(sx < 0, 0.0, t)
        sx = np.maximum(sx, 0)
        t = np.where(sx >= n_in - 1, 0.0, t)
        sx = np.minimum(sx, n_in - 1)
        idx = np.stack([sx, sx + 1], axis=1)
        w = np.stack([1 - t, t], axis=1)
    elif interpolation == cv2.INTER_CUBIC:
        idx = sx[:, None] + np.arange(-1, 3)[None, :]
        w = _cubic_w(t)
    elif interpolation == cv2.INTER_LANCZOS4:
        idx = sx[:, None] + np.arange(-3, 5)[None, :]
        w = _lanczos_w(t)
    else:
        raise ValueError(f"unsupported interpolation {interpolation}")
    return np.clip(idx, 0, n_in - 1), w


def _apply_taps(arr, axis, idx, w, base):
    """Weighted gather along axis; fixed tap order so the result is chunk-invariant."""
    arr = np.moveaxis(arr, axis, -1)
    out = np.zeros(arr.shape[:-1] + (idx.shape[0],), dtype=np.float64)
    for k in range(idx.shape[1]):
        out += arr[..., idx[:, k] - base].astype(np.float64) * w[:, k]
    return np.moveaxis(out, -1, axis)


def _cast_like(x, dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(x), info.min, info.max).astype(dtype)
    return x.astype(dtype)


def _resize_chunk_taps(input_data, out_start, out_end, input_shape, output_shape,
                       interpolation):
    shrink = [ni >= no for ni, no in zip(input_shape, output_shape)]
    # 2D cv2 calls: (y, x) for the YX pass, (z, x-at-target-size) for the Z pass.
    area_flags = [shrink[0], shrink[1] and shrink[2], shrink[1] and shrink[2]]
    taps = [
        _axis_taps(o0, o1, ni, no, interpolation, flag)
        for o0, o1, ni, no, flag in zip(out_start, out_end, input_shape,
                                        output_shape, area_flags)
    ]
    lo = [int(i.min()) for i, _ in taps]
    hi = [int(i.max()) + 1 for i, _ in taps]
    src = input_data[tuple(slice(a, b) for a, b in zip(lo, hi))]
    if not src.any():
        return None
    dtype = src.dtype
    if interpolation == cv2.INTER_NEAREST:
        return src[np.ix_(*(i[:, 0] - b for (i, _), b in zip(taps, lo)))]
    # Same pass order and intermediate rounding as _resize_chunk_3d: YX, then Z.
    yx = _apply_taps(src, 2, taps[2][0], taps[2][1], lo[2])
    yx = _cast_like(_apply_taps(yx, 1, taps[1][0], taps[1][1], lo[1]), dtype)
    if input_shape[0] == output_shape[0]:
        return yx
    return _cast_like(_apply_taps(yx, 0, taps[0][0], taps[0][1], lo[0]), dtype)


def _resize_chunk_worker(args: Tuple) -> Optional[Tuple[int, int, int]]:
    """Process a single output chunk.

    The result for every output voxel depends only on global coordinates, so it
    is independent of the chunk grid. When the reduced input/output ratio is
    small on every axis and the mode is nearest/linear/area, the chunk is computed with cv2.resize on a
    ratio-aligned, halo-padded window (bit-identical to a whole-volume
    _resize_chunk_3d). Otherwise cv2's sampling is reproduced with explicit
    float64 taps (chunk-invariant, within rounding of cv2's fixed-point maths).
    """
    (
        input_path,
        output_path,
        resolution,
        oz_start,
        oy_start,
        ox_start,
        out_chunk_size,
        input_shape,
        output_shape,
        interpolation,
        output_dtype,
    ) = args

    input_store = zarr.open(input_path, mode="r")
    output_store = zarr.open(output_path, mode="r+")
    input_data = input_store[resolution]
    output_data = output_store[resolution]

    out_start = (oz_start, oy_start, ox_start)
    out_end = tuple(
        min(s + c, n) for s, c, n in zip(out_start, out_chunk_size, output_shape)
    )

    exact = interpolation in _EXACT_MODES and all(
        _axis_block(ni, no)[0] <= _MAX_EXACT_BLOCK
        for ni, no in zip(input_shape, output_shape)
    )
    fn = _resize_chunk_exact if exact else _resize_chunk_taps
    resized = fn(input_data, out_start, out_end, tuple(input_shape),
                 tuple(output_shape), interpolation)
    if resized is None:
        return None

    if output_dtype is not None and resized.dtype != output_dtype:
        info = (
            np.iinfo(output_dtype)
            if np.issubdtype(output_dtype, np.integer)
            else np.finfo(output_dtype)
        )
        resized = np.clip(resized, info.min, info.max).astype(output_dtype)

    output_data[oz_start:out_end[0], oy_start:out_end[1], ox_start:out_end[2]] = resized
    return (oz_start, oy_start, ox_start)


@register_task("resize")
class ResizeTask(ZarrTask):
    """Resize an OME-Zarr to match the shape of a reference OME-Zarr."""

    def __init__(self, config: ResizeConfig):
        super().__init__(config)
        self.config: ResizeConfig = config
        self._resolutions: List[str] = []
        self._interpolation: int = cv2.INTER_LINEAR
        self._current_resolution: str = "0"
        self._input_shape: Tuple[int, ...] = ()
        self._target_shape: Tuple[int, ...] = ()
        self._out_chunks: Tuple[int, ...] = ()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add resize-specific arguments."""
        parser.add_argument(
            "--reference",
            type=str,
            required=True,
            help="Path to the reference OME-Zarr file (target shape)",
        )
        parser.add_argument(
            "--interpolation",
            type=str,
            default="linear",
            choices=["nearest", "linear", "cubic", "area", "lanczos"],
            help="Interpolation method (default: linear)",
        )
        parser.add_argument(
            "--output-dtype",
            type=str,
            default=None,
            help="Output dtype (e.g., uint8). If not specified, uses input dtype.",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ResizeTask":
        """Create task from parsed arguments."""
        # Parse output dtype if specified
        output_dtype = np.dtype(args.output_dtype) if args.output_dtype else None

        base_config = make_task_config(args)
        config = ResizeConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            reference_zarr=args.reference,
            interpolation=args.interpolation,
            output_dtype=output_dtype,
        )
        return cls(config)

    def _get_resolutions(self, zarr_path: str) -> List[str]:
        """Get list of resolution levels from OME-Zarr."""
        store = zarr.open(zarr_path, mode="r")

        # Try to read from .zattrs
        zattrs_path = Path(zarr_path) / ".zattrs"
        if zattrs_path.exists():
            with open(zattrs_path) as f:
                attrs = json.load(f)
            if "multiscales" in attrs:
                datasets = attrs["multiscales"][0]["datasets"]
                return [d["path"] for d in datasets]

        # Fallback: find numeric directories
        resolutions = []
        for item in sorted(Path(zarr_path).iterdir()):
            if item.is_dir() and item.name.isdigit():
                resolutions.append(item.name)
        return resolutions

    def prepare(self) -> None:
        """Get resolutions and set up interpolation."""
        # Map interpolation names to cv2 constants
        interpolation_map = {
            "nearest": cv2.INTER_NEAREST,
            "linear": cv2.INTER_LINEAR,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
            "lanczos": cv2.INTER_LANCZOS4,
        }
        self._interpolation = interpolation_map[self.config.interpolation]

        # Get resolutions from input
        self._resolutions = self._get_resolutions(self.config.input_zarr)
        print(f"Found {len(self._resolutions)} resolution levels: {self._resolutions}")

        # Create output directory
        output_path = Path(self.config.output_zarr)
        output_path.mkdir(parents=True, exist_ok=True)

        # Copy metadata files from input
        input_path = Path(self.config.input_zarr)
        for meta_file in [".zattrs", ".zgroup"]:
            src = input_path / meta_file
            if src.exists():
                shutil.copy(src, output_path / meta_file)
                print(f"Copied {meta_file}")

    def run(self) -> None:
        """Execute resize for all resolution levels."""
        self.prepare()

        from concurrent.futures import ProcessPoolExecutor, as_completed

        from tqdm import tqdm

        output_path = Path(self.config.output_zarr)

        for resolution in self._resolutions:
            print(f"\n{'='*60}")
            print(f"Processing resolution level: {resolution}")
            print(f"{'='*60}")

            self._current_resolution = resolution

            # Open input and reference
            input_store = zarr.open(self.config.input_zarr, mode="r")
            ref_store = zarr.open(self.config.reference_zarr, mode="r")

            input_data = input_store[resolution]
            ref_data = ref_store[resolution]

            self._input_shape = input_data.shape
            self._target_shape = ref_data.shape

            print(f"Input shape: {self._input_shape}")
            print(f"Target shape: {self._target_shape}")
            print(f"Input dtype: {input_data.dtype}")
            print(
                f"Output dtype: {self.config.output_dtype if self.config.output_dtype else input_data.dtype}"
            )

            # Get compression and chunks from input
            compressor = input_data.compressor
            input_chunks = input_data.chunks

            # Use input chunks, but cap at target shape
            self._out_chunks = tuple(
                min(c, s) for c, s in zip(input_chunks, self._target_shape)
            )

            print(f"Compressor: {compressor}")
            print(f"Output chunks: {self._out_chunks}")

            # Create output array for this resolution
            output_store = zarr.open(str(output_path), mode="a")
            output_store.create_dataset(
                resolution,
                shape=self._target_shape,
                chunks=self._out_chunks,
                dtype=self.config.output_dtype
                if self.config.output_dtype
                else input_data.dtype,
                compressor=compressor,
                dimension_separator="/",
                overwrite=True,
            )

            # Generate chunk tasks
            work_items = list(self._generate_level_work_items(resolution))
            total_chunks = len(work_items)
            print(
                f"Processing {total_chunks} chunks with {self.config.num_workers} workers..."
            )

            # Process chunks in parallel
            with ProcessPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {
                    executor.submit(_resize_chunk_worker, arg): arg
                    for arg in work_items
                }

                with tqdm(total=total_chunks, desc=f"Resolution {resolution}") as pbar:
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Error processing chunk: {e}")
                            raise
                        pbar.update(1)

        self.finalize()

    def _generate_level_work_items(self, resolution: str) -> Iterable[Any]:
        """Generate work items for a specific resolution level."""
        for oz in range(0, self._target_shape[0], self._out_chunks[0]):
            for oy in range(0, self._target_shape[1], self._out_chunks[1]):
                for ox in range(0, self._target_shape[2], self._out_chunks[2]):
                    yield (
                        self.config.input_zarr,
                        self.config.output_zarr,
                        resolution,
                        oz,
                        oy,
                        ox,
                        self._out_chunks,
                        self._input_shape,
                        self._target_shape,
                        self._interpolation,
                        self.config.output_dtype,
                    )

    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items - handled in custom run()."""
        return []

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _resize_chunk_worker(args)

    def finalize(self) -> None:
        """Print completion message."""
        print(f"\nDone! Output saved to {self.config.output_zarr}")
