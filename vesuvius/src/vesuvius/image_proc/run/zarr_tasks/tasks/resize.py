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


def _resize_chunk_worker(args: Tuple) -> Optional[Tuple[int, int, int]]:
    """Process a single output chunk."""
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

    # Open zarr in each worker
    input_store = zarr.open(input_path, mode="r")
    output_store = zarr.open(output_path, mode="r+")

    input_data = input_store[resolution]
    output_data = output_store[resolution]

    # Calculate output chunk bounds
    oz_end = min(oz_start + out_chunk_size[0], output_shape[0])
    oy_end = min(oy_start + out_chunk_size[1], output_shape[1])
    ox_end = min(ox_start + out_chunk_size[2], output_shape[2])

    out_chunk_shape = (oz_end - oz_start, oy_end - oy_start, ox_end - ox_start)

    # Calculate corresponding input region (with float precision)
    scale_z = input_shape[0] / output_shape[0]
    scale_y = input_shape[1] / output_shape[1]
    scale_x = input_shape[2] / output_shape[2]

    iz_start = int(oz_start * scale_z)
    iy_start = int(oy_start * scale_y)
    ix_start = int(ox_start * scale_x)

    iz_end = min(int(np.ceil(oz_end * scale_z)), input_shape[0])
    iy_end = min(int(np.ceil(oy_end * scale_y)), input_shape[1])
    ix_end = min(int(np.ceil(ox_end * scale_x)), input_shape[2])

    # Ensure we have at least 1 pixel
    if iz_end <= iz_start:
        iz_end = iz_start + 1
    if iy_end <= iy_start:
        iy_end = iy_start + 1
    if ix_end <= ix_start:
        ix_end = ix_start + 1

    # Read input region
    input_chunk = input_data[iz_start:iz_end, iy_start:iy_end, ix_start:ix_end]

    # Skip empty chunks (all zeros)
    if not input_chunk.any():
        return None

    # Resize to output chunk shape
    resized = _resize_chunk_3d(input_chunk, out_chunk_shape, interpolation)

    # Convert dtype if specified
    if output_dtype is not None and resized.dtype != output_dtype:
        # Clip to valid range and cast
        info = (
            np.iinfo(output_dtype)
            if np.issubdtype(output_dtype, np.integer)
            else np.finfo(output_dtype)
        )
        resized = np.clip(resized, info.min, info.max).astype(output_dtype)

    # Write to output
    output_data[oz_start:oz_end, oy_start:oy_end, ox_start:ox_end] = resized

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
