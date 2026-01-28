"""Export TIFFs task for zarr arrays.

Extracts non-overlapping chunks from an OME-Zarr and saves as 3D TIFFs.
"""

from __future__ import annotations

import argparse
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import tifffile
import zarr

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task


# Thread-local storage for zarr stores
_thread_local = threading.local()
_zarr_path: Optional[str] = None
_zarr2_path: Optional[str] = None
_resolution: Optional[str] = None


@dataclass
class ExportTifsConfig(TaskConfig):
    """Configuration for export TIFFs task."""

    resolution: str = "0"
    chunk_size: int = 64
    zarr2_path: Optional[str] = None
    out2_dir: Optional[str] = None
    ignore_label: Optional[int] = None
    bg_dir: Optional[str] = None
    bbox: Optional[str] = None


def _parse_bbox(bbox_str: str, shape: Tuple[int, int, int]) -> Tuple[Tuple[int, int], ...]:
    """Parse a numpy-like slice string into bounds.

    Args:
        bbox_str: String in format 'z_start:z_end,y_start:y_end,x_start:x_end'
        shape: Tuple of (z_size, y_size, x_size) for default bounds

    Returns:
        Tuple of ((z_start, z_end), (y_start, y_end), (x_start, x_end))
    """
    parts = bbox_str.split(",")
    if len(parts) != 3:
        raise ValueError(f"bbox must have exactly 3 dimensions (zyx), got {len(parts)}")

    bounds = []
    for i, (part, dim_size) in enumerate(zip(parts, shape)):
        part = part.strip()
        if ":" not in part:
            raise ValueError(
                f"Each dimension must use slice notation (e.g., '0:100'), got '{part}'"
            )

        start_str, end_str = part.split(":", 1)
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else dim_size

        # Handle negative indices like numpy
        if start < 0:
            start = max(0, dim_size + start)
        if end < 0:
            end = max(0, dim_size + end)

        # Clamp to valid range
        start = max(0, min(start, dim_size))
        end = max(0, min(end, dim_size))

        if start >= end:
            raise ValueError(
                f"Invalid slice for dimension {i}: start ({start}) >= end ({end})"
            )

        bounds.append((start, end))

    return tuple(bounds)


def _get_zarr_stores():
    """Get or initialize zarr stores for the current thread."""
    global _zarr_path, _zarr2_path, _resolution
    if not hasattr(_thread_local, "zarr_data1"):
        store1 = zarr.open(_zarr_path, mode="r")
        _thread_local.zarr_data1 = store1[_resolution]
        if _zarr2_path is not None:
            store2 = zarr.open(_zarr2_path, mode="r")
            _thread_local.zarr_data2 = store2[_resolution]
        else:
            _thread_local.zarr_data2 = None
    return _thread_local.zarr_data1, _thread_local.zarr_data2


def _export_chunk_worker(args: Tuple) -> Optional[Path]:
    """Save a single chunk as a TIFF file."""
    (
        z_start,
        y_start,
        x_start,
        chunk_size,
        output_dir,
        output_dir2,
        ignore_label,
        bg_dir,
        bbox_ends,
    ) = args

    # Get thread-local zarr stores
    zarr_data1, zarr_data2 = _get_zarr_stores()

    # Clip to both zarr shape and bbox bounds (if provided)
    z_max, y_max, x_max = zarr_data1.shape[:3]
    if bbox_ends is not None:
        z_max = min(z_max, bbox_ends[0])
        y_max = min(y_max, bbox_ends[1])
        x_max = min(x_max, bbox_ends[2])

    z_end = min(z_start + chunk_size, z_max)
    y_end = min(y_start + chunk_size, y_max)
    x_end = min(x_start + chunk_size, x_max)

    chunk_data = zarr_data1[z_start:z_end, y_start:y_end, x_start:x_end]

    # Skip empty chunks
    if chunk_data.size == 0:
        return None

    # Use count_nonzero which is faster than (arr == 0).all()
    nonzero_count = np.count_nonzero(chunk_data)

    # Skip chunks that contain only zeros
    if nonzero_count == 0:
        return None

    # Check for ignore label if provided
    is_background = False
    if ignore_label is not None:
        is_ignore = chunk_data == ignore_label
        ignore_count = np.count_nonzero(is_ignore)

        # Skip chunks that contain only the ignore label
        if ignore_count == chunk_data.size:
            return None

        # Check if chunk is background (contains only zeros and ignore label)
        if nonzero_count == ignore_count:
            has_zeros = nonzero_count < chunk_data.size
            has_ignore = ignore_count > 0
            is_background = has_zeros and has_ignore

    if is_background:
        if bg_dir is None:
            return None
        target_dir = bg_dir
    else:
        target_dir = output_dir

    output_filename = f"{chunk_size}_z{z_start}_y{y_start}_x{x_start}.tif"
    output_path = Path(target_dir) / output_filename

    # Use ZSTD compression
    tifffile.imwrite(
        output_path,
        chunk_data,
        compression="zstd",
        photometric="minisblack",
    )

    # Process second zarr if provided
    if zarr_data2 is not None:
        chunk_data2 = zarr_data2[z_start:z_end, y_start:y_end, x_start:x_end]

        output_path2 = Path(output_dir2) / output_filename
        tifffile.imwrite(
            output_path2,
            chunk_data2,
            compression="zstd",
            photometric="minisblack",
        )

    return output_path


@register_task("export-tifs")
class ExportTifsTask(ZarrTask):
    """Extract non-overlapping chunks from OME-Zarr and save as 3D TIFFs."""

    def __init__(self, config: ExportTifsConfig):
        super().__init__(config)
        self.config: ExportTifsConfig = config
        self._shape: Tuple[int, int, int] = (0, 0, 0)
        self._bbox_bounds: Optional[Tuple[Tuple[int, int], ...]] = None
        self._saved_count: int = 0

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add export-tifs-specific arguments."""
        parser.add_argument(
            "--resolution",
            type=str,
            default="0",
            help="Resolution level to extract from (e.g., '0', '1', '2')",
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            required=True,
            help="Chunk size (single int applied to z, y, x)",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            required=True,
            help="Output directory for TIFF files",
        )
        parser.add_argument(
            "--zarr2",
            type=str,
            default=None,
            help="Path to a second OME-Zarr file to extract chunks from the same positions",
        )
        parser.add_argument(
            "--out2",
            type=str,
            default=None,
            help="Output directory for second zarr's TIFF files (required if --zarr2 is provided)",
        )
        parser.add_argument(
            "--ignore-label",
            type=int,
            default=None,
            help="Skip chunks where the first zarr contains only this label value",
        )
        parser.add_argument(
            "--bg-dir",
            type=str,
            default=None,
            help="Output directory for background chunks (containing only zeros and ignore label)",
        )
        parser.add_argument(
            "--bbox",
            type=str,
            default=None,
            help="3D bounding box in numpy slice format (zyx): 'z_start:z_end,y_start:y_end,x_start:x_end'",
        )

    @classmethod
    def validate_args(cls, args: argparse.Namespace) -> None:
        """Validate export-tifs arguments."""
        if args.zarr2 is not None and args.out2 is None:
            raise SystemExit("Error: --out2 is required when --zarr2 is provided")

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ExportTifsTask":
        """Create task from parsed arguments."""
        cls.validate_args(args)

        base_config = make_task_config(args)
        config = ExportTifsConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=args.output_dir,  # Use output_dir as output_zarr
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            resolution=getattr(args, "resolution", "0"),
            chunk_size=args.chunk_size,
            zarr2_path=args.zarr2,
            out2_dir=args.out2,
            ignore_label=args.ignore_label,
            bg_dir=args.bg_dir,
            bbox=args.bbox,
        )
        return cls(config)

    def prepare(self) -> None:
        """Create output directories and set up global state."""
        global _zarr_path, _zarr2_path, _resolution

        # Create output directories
        output_dir = Path(self.config.output_zarr)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.out2_dir is not None:
            output_dir2 = Path(self.config.out2_dir)
            output_dir2.mkdir(parents=True, exist_ok=True)

        if self.config.bg_dir is not None:
            bg_dir = Path(self.config.bg_dir)
            bg_dir.mkdir(parents=True, exist_ok=True)

        # Open zarr to get dimensions
        store = zarr.open(self.config.input_zarr, mode="r")
        data = store[self.config.resolution]

        self._shape = data.shape[:3]

        # Parse bounding box if provided
        if self.config.bbox is not None:
            self._bbox_bounds = _parse_bbox(self.config.bbox, self._shape)

        print(f"Data shape at resolution '{self.config.resolution}': {data.shape}")
        print(f"Chunk size: {self.config.chunk_size}")
        if self.config.bbox is not None:
            (z_start, z_end), (y_start, y_end), (x_start, x_end) = self._bbox_bounds
            print(f"Bounding box (zyx): [{z_start}:{z_end}, {y_start}:{y_end}, {x_start}:{x_end}]")
        print(f"Output directory: {output_dir}")
        if self.config.zarr2_path is not None:
            print(f"Second zarr: {self.config.zarr2_path}")
            print(f"Second output directory: {self.config.out2_dir}")
        if self.config.ignore_label is not None:
            print(f"Ignoring chunks containing only label: {self.config.ignore_label}")
        if self.config.bg_dir is not None:
            print(f"Background directory: {self.config.bg_dir}")

        # Set global variables for thread-local initialization
        _zarr_path = self.config.input_zarr
        _zarr2_path = self.config.zarr2_path
        _resolution = self.config.resolution

    def generate_work_items(self) -> Iterable[Any]:
        """Generate chunk coordinates for parallel processing."""
        chunk_size = self.config.chunk_size

        # Determine bounds
        if self._bbox_bounds is not None:
            (z_start_bound, z_end_bound), (y_start_bound, y_end_bound), (x_start_bound, x_end_bound) = self._bbox_bounds
            bbox_ends = (z_end_bound, y_end_bound, x_end_bound)
        else:
            z_start_bound, z_end_bound = 0, self._shape[0]
            y_start_bound, y_end_bound = 0, self._shape[1]
            x_start_bound, x_end_bound = 0, self._shape[2]
            bbox_ends = None

        for z_start in range(z_start_bound, z_end_bound, chunk_size):
            for y_start in range(y_start_bound, y_end_bound, chunk_size):
                for x_start in range(x_start_bound, x_end_bound, chunk_size):
                    yield (
                        z_start,
                        y_start,
                        x_start,
                        chunk_size,
                        self.config.output_zarr,
                        self.config.out2_dir,
                        self.config.ignore_label,
                        self.config.bg_dir,
                        bbox_ends,
                    )

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _export_chunk_worker(args)

    def run(self) -> None:
        """Execute export using ThreadPoolExecutor for I/O-bound work."""
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from tqdm import tqdm

        self.prepare()

        work_items = list(self.generate_work_items())
        total_chunks = len(work_items)
        print(f"Total chunks to process: {total_chunks}")

        # Process chunks using threads (zarr releases GIL during I/O)
        saved_count = 0
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = {
                executor.submit(_export_chunk_worker, coord): coord
                for coord in work_items
            }

            with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        saved_count += 1
                    pbar.update(1)

        self._saved_count = saved_count
        self.finalize()

    def finalize(self) -> None:
        """Print completion summary."""
        results = getattr(self, "_results", [])
        if results:
            saved_count = sum(1 for r in results if r is not None)
            total_chunks = len(results)
        else:
            saved_count = self._saved_count
            total_chunks = len(list(self.generate_work_items())) if saved_count == 0 else "?"

        print(f"Done! Saved {saved_count}/{total_chunks} chunks to {self.config.output_zarr}")
