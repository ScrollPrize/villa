"""Remap task for zarr arrays.

Remaps values in an OME-Zarr array chunk-wise using a dictionary mapping.
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import zarr

from ..base import TaskConfig, ZarrTask, make_task_config
from ..registry import register_task
from ..utils import get_chunk_slices

# Try to import fastremap for efficient remapping
try:
    import fastremap

    HAS_FASTREMAP = True
except ImportError:
    HAS_FASTREMAP = False


@dataclass
class RemapConfig(TaskConfig):
    """Configuration for remap task."""

    remap_dict: Dict[int, int] = field(default_factory=dict)
    default_to: Optional[int] = None
    all_levels: bool = True


def _remap_chunk_worker(args: Tuple) -> int:
    """Process a single chunk: read, remap values, write."""
    (
        chunk_idx,
        chunk_slice,
        input_path,
        output_path,
        remap_dict,
        resolution_level,
        default_to,
    ) = args

    # Open stores (each worker opens its own handle)
    input_store = zarr.open(input_path, mode="r")
    output_store = zarr.open(output_path, mode="r+")

    input_arr = input_store[str(resolution_level)]
    output_arr = output_store[str(resolution_level)]

    # Read chunk data
    data = input_arr[chunk_slice]

    # If default_to is set, extend remap_dict to map all unique values not already mapped
    if default_to is not None:
        unique_values = np.unique(data)
        remap_dict = remap_dict.copy()  # Don't mutate the original
        for v in unique_values:
            if v not in remap_dict:
                remap_dict[v] = default_to

    # Apply remapping
    if HAS_FASTREMAP:
        remapped = fastremap.remap(
            data, remap_dict, preserve_missing_labels=True, in_place=False
        )
    else:
        # Fallback using numpy
        remapped = data.copy()
        for old_val, new_val in remap_dict.items():
            remapped[data == old_val] = new_val

    # Write to output
    output_arr[chunk_slice] = remapped

    return chunk_idx


@register_task("remap")
class RemapTask(ZarrTask):
    """Remap values in a zarr array using a dictionary mapping."""

    def __init__(self, config: RemapConfig):
        super().__init__(config)
        self.config: RemapConfig = config
        self._levels_to_process: List[int] = []
        self._current_level: int = 0

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Add remap-specific arguments."""
        parser.add_argument(
            "--remap",
            type=str,
            action="append",
            help="Value remapping in format 'old:new' (can be specified multiple times)",
        )
        parser.add_argument(
            "--default-to",
            type=int,
            default=None,
            dest="default_to",
            help="Map all values not explicitly remapped to this value",
        )
        parser.add_argument(
            "--all-levels",
            action="store_true",
            default=True,
            help="Process all resolution levels (default: True)",
        )

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "RemapTask":
        """Create task from parsed arguments."""
        # Parse remap arguments into dictionary
        remap_dict = {}
        if args.remap:
            for mapping in args.remap:
                try:
                    old_str, new_str = mapping.split(":")
                    remap_dict[int(old_str)] = int(new_str)
                except ValueError:
                    raise SystemExit(
                        f"Error: Invalid remap format '{mapping}'. Use 'old:new'"
                    )

        if not remap_dict:
            raise SystemExit(
                "Error: At least one --remap mapping is required. "
                "Example: --remap 100:2 --remap 200:3"
            )

        base_config = make_task_config(args)
        config = RemapConfig(
            input_zarr=base_config.input_zarr,
            output_zarr=base_config.output_zarr,
            num_workers=base_config.num_workers,
            inplace=base_config.inplace,
            level=base_config.level,
            remap_dict=remap_dict,
            default_to=args.default_to,
            all_levels=getattr(args, "all_levels", True),
        )
        return cls(config)

    def prepare(self) -> None:
        """Detect levels and create output zarr."""
        print(f"Input: {self.config.input_zarr}")
        print(f"Output: {self.config.output_zarr}")
        print(f"Workers: {self.config.num_workers}")
        print(f"Remap values: {self.config.remap_dict}")
        if self.config.default_to is not None:
            print(f"Default unmapped values to: {self.config.default_to}")

        if not HAS_FASTREMAP:
            print("Warning: fastremap not installed, using numpy (slower)")

        # Open input to get metadata
        input_store = zarr.open(self.config.input_zarr, mode="r")

        # Determine which levels to process
        if self.config.level is not None:
            self._levels_to_process = [self.config.level]
        elif self.config.all_levels:
            # Find all numeric keys (resolution levels)
            self._levels_to_process = sorted(
                [int(k) for k in input_store.keys() if k.isdigit()]
            )
            print(f"Found {len(self._levels_to_process)} resolution levels: {self._levels_to_process}")
        else:
            self._levels_to_process = [0]

        # Create output zarr
        output_path = Path(self.config.output_zarr)
        if output_path.exists():
            print("Output already exists, opening in r+ mode")
        else:
            print("Creating output zarr")
            zarr.open(self.config.output_zarr, mode="w")

    def run(self) -> None:
        """Execute remap for all levels."""
        self.prepare()

        input_store = zarr.open(self.config.input_zarr, mode="r")
        output_store = zarr.open(self.config.output_zarr, mode="a")

        for level in self._levels_to_process:
            print(f"\n{'='*60}")
            print(f"Processing level {level}")
            print(f"{'='*60}")

            self._current_level = level
            input_arr = input_store[str(level)]

            print(f"Array shape: {input_arr.shape}")
            print(f"Array chunks: {input_arr.chunks}")
            print(f"Array dtype: {input_arr.dtype}")

            # Create output array with same properties
            if str(level) not in output_store:
                output_store.create_dataset(
                    str(level),
                    shape=input_arr.shape,
                    chunks=input_arr.chunks,
                    dtype=input_arr.dtype,
                    compressor=input_arr.compressor,
                )

            # Generate and process work items for this level
            work_items = list(self._generate_level_work_items(level, input_arr))
            print(f"Total chunks to process: {len(work_items)}")

            from multiprocessing import Pool

            from tqdm import tqdm

            with Pool(processes=self.config.num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap(_remap_chunk_worker, work_items),
                        total=len(work_items),
                        desc=f"Level {level}",
                    )
                )

            print(f"Completed processing {len(results)} chunks for level {level}")

        self.finalize()

    def _generate_level_work_items(
        self, level: int, input_arr: zarr.Array
    ) -> Iterable[Any]:
        """Generate work items for a specific level."""
        chunk_slices = get_chunk_slices(input_arr.shape, input_arr.chunks)

        for idx, chunk_slice in enumerate(chunk_slices):
            yield (
                idx,
                chunk_slice,
                self.config.input_zarr,
                self.config.output_zarr,
                self.config.remap_dict,
                level,
                self.config.default_to,
            )

    def generate_work_items(self) -> Iterable[Any]:
        """Generate work items - handled in custom run()."""
        return []

    @staticmethod
    def process_item(args: Any) -> Any:
        """Process a single chunk."""
        return _remap_chunk_worker(args)

    def finalize(self) -> None:
        """Copy metadata to output."""
        input_store = zarr.open(self.config.input_zarr, mode="r")
        output_store = zarr.open(self.config.output_zarr, mode="a")

        # Copy metadata if present
        if ".zattrs" in input_store:
            output_store.attrs.update(input_store.attrs)

        # Copy metadata.json if it exists
        metadata_src = Path(self.config.input_zarr) / "metadata.json"
        if metadata_src.exists():
            metadata_dst = Path(self.config.output_zarr) / "metadata.json"
            shutil.copy2(metadata_src, metadata_dst)
            print("Copied metadata.json")

        print("\nDone!")
