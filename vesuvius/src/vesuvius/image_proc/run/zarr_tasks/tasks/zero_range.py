"""Zero-range task for zarr arrays.

Zeros out all chunks within a specified z-range, operating in-place.
Supports OME-Zarr with automatic z-range scaling for pyramid levels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Iterable, List, Tuple

import numpy as np
import zarr
from tqdm import tqdm

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task
from ..utils import get_chunk_coords


@dataclass
class ZeroRangeConfig(TaskConfig):
    """Configuration for zero-range task."""

    z_start: int = 0
    z_end: int = 0
    replace_value: int = 0


def _zero_range_worker(args: Tuple) -> Tuple[Tuple[int, int], ...]:
    """Multiprocessing worker for filling chunks in a z-range with a value.

    Args:
        args: Tuple of (zarr_path, chunk_coords, z_start, z_end, replace_value)

    Returns:
        The chunk coordinates that were processed
    """
    zarr_path, chunk_coords, z_start, z_end, replace_value = args
    arr = zarr.open(zarr_path, mode="r+")

    # chunk_coords is ((z_start, z_stop), (y_start, y_stop), (x_start, x_stop))
    cz_start, cz_stop = chunk_coords[0]

    # Calculate overlap
    zero_start = max(z_start, cz_start)
    zero_stop = min(z_end + 1, cz_stop)  # +1 for inclusive end

    # Build full slices
    slices = (slice(zero_start, zero_stop),) + tuple(
        slice(start, stop) for start, stop in chunk_coords[1:]
    )

    # Fill the region with replace_value
    shape = tuple(s.stop - s.start for s in slices)
    arr[slices] = np.full(shape, replace_value, dtype=arr.dtype)

    return chunk_coords


@register_task("zero-range")
class ZeroRangeTask(ZarrTask):
    """Zero out chunks within a specified z-range."""

    def __init__(self, config: ZeroRangeConfig):
        super().__init__(config)
        self.config: ZeroRangeConfig = config

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add zero-range-specific arguments."""
        parser.add_argument(
            "--z-start",
            type=int,
            required=True,
            help="Start of z-range to fill (inclusive)",
        )
        parser.add_argument(
            "--z-end",
            type=int,
            required=True,
            help="End of z-range to fill (inclusive)",
        )
        parser.add_argument(
            "--replace-value",
            type=int,
            default=0,
            help="Value to fill the z-range with (default: 0)",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "ZeroRangeTask":
        """Create task from parsed arguments."""
        base_config = make_task_config(args)
        config = ZeroRangeConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            z_start=args.z_start,
            z_end=args.z_end,
            replace_value=args.replace_value,
        )
        return cls(config)

    @classmethod
    def validate_args(cls, args: argparse.Namespace) -> None:
        """Validate z-range arguments."""
        if args.z_start > args.z_end:
            raise SystemExit(
                f"Error: --z-start ({args.z_start}) must be <= --z-end ({args.z_end})"
            )

    def prepare(self) -> None:
        """Detect OME-Zarr structure and prepare for processing."""
        self._is_ome_zarr, all_levels = self._detect_ome_zarr(self.config.input_zarr)

        if self._is_ome_zarr:
            if self.config.level is not None:
                self._levels_to_process = [str(self.config.level)]
                if str(self.config.level) not in all_levels:
                    raise SystemExit(
                        f"Error: Level {self.config.level} not found. "
                        f"Available levels: {all_levels}"
                    )
            else:
                self._levels_to_process = all_levels

            print(
                f"Detected OME-Zarr with {len(all_levels)} resolution levels: {all_levels}"
            )
            if len(self._levels_to_process) > 1:
                print(f"Will process levels: {self._levels_to_process}")
        else:
            self._levels_to_process = [None]

        input_z = self._get_input_array(
            self._levels_to_process[0] if self._is_ome_zarr else None
        )

        print(f"Input array shape: {input_z.shape}")
        print(f"Input array chunks: {input_z.chunks}")
        print(f"Input array dtype: {input_z.dtype}")
        print(f"Task: zero-range | z=[{self.config.z_start}, {self.config.z_end}] | replace_value={self.config.replace_value}")

    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items - handled in custom run() for zero-range."""
        # Zero-range uses custom run() method
        return []

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _zero_range_worker(args)

    def run(self) -> None:
        """Execute zero-range with multi-level z-scaling."""
        self.prepare()

        for level_idx, level in enumerate(self._levels_to_process):
            if self._is_ome_zarr:
                level_path = f"{self.config.input_zarr}/{level}"
                level_int = int(level)
                print(f"\nProcessing level {level}...")
            else:
                level_path = self.config.input_zarr
                level_int = 0

            # Scale z-range by 2^level (matches pyramid downsampling)
            scale_factor = 2**level_int
            scaled_z_start = self.config.z_start // scale_factor
            scaled_z_end = self.config.z_end // scale_factor

            arr = zarr.open(level_path, mode="r")
            print(f"  Shape: {arr.shape}, Chunks: {arr.chunks}")
            print(f"  Z-range: [{scaled_z_start}, {scaled_z_end}]")

            # Get all chunk coordinates
            all_chunk_coords = get_chunk_coords(arr.shape, arr.chunks)

            # Filter to only chunks overlapping the z-range
            # Overlap condition: z_start <= chunk_z_stop - 1 and z_end >= chunk_z_start
            filtered_chunks = []
            for coords in all_chunk_coords:
                cz_start, cz_stop = coords[0]
                if scaled_z_start <= cz_stop - 1 and scaled_z_end >= cz_start:
                    filtered_chunks.append(coords)

            if not filtered_chunks:
                print(f"  No chunks overlap z-range, skipping level {level}")
                continue

            print(f"  Chunks to process: {len(filtered_chunks)}")

            # Build work items
            work_items = [
                (level_path, coords, scaled_z_start, scaled_z_end, self.config.replace_value)
                for coords in filtered_chunks
            ]

            desc = (
                f"Zeroing level {level}"
                if self._is_ome_zarr
                else "Zeroing z-range"
            )

            with Pool(processes=self.config.num_workers) as pool:
                for _ in tqdm(
                    pool.imap_unordered(_zero_range_worker, work_items),
                    total=len(work_items),
                    desc=desc,
                ):
                    pass

        self.finalize()

    def finalize(self) -> None:
        """Print completion message."""
        print(f"\nZero-range complete: {self.config.input_zarr}")
        print(f"Filled z-range: [{self.config.z_start}, {self.config.z_end}] with value {self.config.replace_value}")
